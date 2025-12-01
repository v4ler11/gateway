import asyncio

from typing import List, AsyncGenerator, Optional, Literal

import aiohttp
import pysbd

from core.logger import error
from core.routers.oai.models import ChatCompletionsResponseStreaming, ChatDelta
from core.routers.oai.schemas import AudioPost, ChatPost
from core.routers.oai.sentence_collector import SentenceCollector
from core.routers.utils import parse_sse_streaming
from models.definitions import ModelLLMAny, ModelTTSAny
from models.urls import URLs
from tts.inference.encode_audio_stream import encode_audio_stream


async def stream_with_chat(
        http_session: aiohttp.ClientSession,
        model: ModelLLMAny,
        post: ChatPost,
) -> AsyncGenerator[ChatCompletionsResponseStreaming, None]:
    if not post.stream:
        raise ValueError(f"post.stream should be True, got post.stream={post.stream}")

    async with http_session.post(
        url=model.urls.generate,
        json=post.model_dump(),
    ) as response:
        async for chunk in parse_sse_streaming(response.content):
            if chunk:
                yield ChatCompletionsResponseStreaming.model_validate(chunk)


async def stream_audio(
        http_session: aiohttp.ClientSession,
        model: ModelTTSAny,
        post: AudioPost
) -> AsyncGenerator[bytes, None]:
    assert isinstance(model.record.urls, URLs)
    if not post.stream:
        raise ValueError(f"post.stream should be True, got post.stream={post.stream}")

    try:
        async with http_session.post(
                url=model.record.urls.generate,
                json=post.model_dump()
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                yield error_text.encode()
                return

            async for chunk in response.content.iter_any():
                if chunk:
                    yield chunk

    except Exception as e:  # noqa
        pass


async def stream_with_chat_synthesised(
        http_session: aiohttp.ClientSession,
        tts_model: ModelTTSAny,
        a_post: AudioPost,
        llm_stream: AsyncGenerator[ChatCompletionsResponseStreaming, None],
        segmenter: pysbd.Segmenter,
) -> AsyncGenerator[str | bytes, None]:
    text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def producer():
        collector = SentenceCollector(segmenter=segmenter)

        try:
            async for chunk in llm_stream: # noqa
                if not chunk.choices: continue

                delta = chunk.choices[0].delta
                assert isinstance(delta, ChatDelta)
                if not delta or not delta.content: continue

                sentences = collector.put(delta.content)

                for sentence in sentences:
                    await text_queue.put(sentence)

            remaining = collector.flush()
            for sentence in remaining:
                await text_queue.put(sentence)

        except Exception as e: # noqa
            error(f"Error in LLM stream producer: {e}")
        finally:
            await text_queue.put(None)

    producer_task = asyncio.create_task(producer())
    pending_item: Optional[str] = None

    try:
        while True:
            item = pending_item or await text_queue.get()
            if item is None:
                break

            pending_item = None

            current_batch: List[str] = [item]
            chars_cnt = len(item)
            text_queue.task_done()

            while not text_queue.empty():
                # todo: what if new item in context_size big?
                next_item = None
                try:
                    next_item = text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if next_item is None:
                    pending_item = next_item
                    text_queue.task_done()
                    break

                if chars_cnt + len(next_item) > tts_model.record.context_size * 0.9:
                    pending_item = next_item
                    text_queue.task_done()
                    break

                current_batch.append(next_item)
                chars_cnt += len(next_item)
                text_queue.task_done()

            full_text = " ".join(current_batch)
            try:
                yield full_text

                a_post_clone = a_post.model_copy(update={"text": full_text})
                async for chunk in stream_audio(http_session, tts_model, a_post_clone):
                    yield chunk

            except Exception as e:
                error(f"Error generating audio for batch '{full_text[:30]}...': {e}")

    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

# todo: fix me
#data: {"id":"msg_64497e0db3bc55241d8f9f2c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"audio":{"id":"audio_cb3db002fd7b29a9a0fdc5a7","transcript":"Hello! "}},"finish_reason":null}],"created":1764630238,"model":"gpt-oss-20b+kokoro","system_fingerprint":null}

#data: {"id":"msg_64497e0db3bc55241d8f9f2c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"audio":{"transcript":"I'm just a bundle of code, so I don't have feelings, but I'm here and ready to help you. "}},"finish_reason":null}],"created":1764630238,"model":"gpt-oss-20b+kokoro","system_fingerprint":null}

# gat-core  | 20251201 23:08:38 ERROR [stream_utils.py:92 producer] Error in LLM stream producer:
async def encode_synthesized_stream(
        tts_model: ModelTTSAny,
        synthesizer: AsyncGenerator[str | bytes, None],
        output_format: Literal["pcm", "wav", "mp3", "ogg"],
) -> AsyncGenerator[str | bytes, None]:
    pcm_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    output_queue: asyncio.Queue[Optional[str | bytes]] = asyncio.Queue()

    async def stream_processor():
        has_unprocessed_audio = False
        try:
            async for chunk in synthesizer:
                if isinstance(chunk, str):
                    if has_unprocessed_audio:
                        await pcm_queue.join()
                        has_unprocessed_audio = False

                    await output_queue.put(chunk)
                else:
                    has_unprocessed_audio = True
                    await pcm_queue.put(chunk)

            if has_unprocessed_audio:
                await pcm_queue.join()

        except Exception as e:
            error(f"Error in stream processor: {e}")
        finally:
            await pcm_queue.put(None)

    async def pcm_feeder() -> AsyncGenerator[bytes, None]:
        while True:
            chunk = await pcm_queue.get()
            if chunk is None:
                pcm_queue.task_done()
                break

            yield chunk

            pcm_queue.task_done()

    async def encoder_runner():
        try:
            async for encoded_chunk in encode_audio_stream(
                    input_stream=pcm_feeder(),
                    output_format=output_format,
                    sample_rate=tts_model.record.constants.sample_rate,
                    channels=tts_model.record.constants.channels,
            ):
                await output_queue.put(encoded_chunk)
        except Exception as e:
            error(f"Error in encoder runner: {e}")
        finally:
            await output_queue.put(None)

    processor_task = asyncio.create_task(stream_processor())
    encoder_task = asyncio.create_task(encoder_runner())

    try:
        while True:
            item = await output_queue.get()
            if item is None:
                output_queue.task_done()
                break

            yield item
            output_queue.task_done()

    finally:
        if not processor_task.done(): processor_task.cancel()
        if not encoder_task.done(): encoder_task.cancel()
        await asyncio.gather(processor_task, encoder_task, return_exceptions=True)
