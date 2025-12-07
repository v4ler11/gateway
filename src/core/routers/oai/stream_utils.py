import asyncio

from typing import List, AsyncGenerator, Optional, Literal

import aiohttp
import pysbd

from core.logger import error
from core.routers.oai.models import ChatCompletionsResponseStreaming, ChatDelta
from core.routers.oai.schemas import ChatPost
from core.routers.oai.sentence_collector import SentenceCollector
from core.routers.utils import parse_sse_streaming
from models.definitions import ModelLLMAny, ModelTTSAny
from models.urls import URLs
from tts.inference.encode_audio_stream import encode_audio_stream
from tts.inference.schemas import TTSAudioPost


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
        post: TTSAudioPost
) -> AsyncGenerator[bytes, None]:
    assert isinstance(model.record.urls, URLs)

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
        a_post: TTSAudioPost,
        llm_stream: AsyncGenerator[ChatCompletionsResponseStreaming, None],
        segmenter: pysbd.Segmenter,
) -> AsyncGenerator[str | bytes, None]:
    text_queue: asyncio.Queue[str] = asyncio.Queue()
    SENTINEL = "<|GATEWAY::PRODUCER::STOP|>" # noqa

    async def producer():
        collector = SentenceCollector(segmenter=segmenter)
        try:
            llm_stream_iterator = llm_stream.__aiter__()
            while True:
                llm_resp = await asyncio.wait_for(
                    llm_stream_iterator.__anext__(),
                    timeout=30.
                )

                if not llm_resp.choices: continue

                delta = llm_resp.choices[0].delta
                content = delta.content if isinstance(delta, ChatDelta) else delta.get("content")

                if not content:
                    continue

                sentences = collector.put(content)
                for sentence in sentences:
                    await text_queue.put(sentence)

        except StopAsyncIteration:
            pass
        except Exception as e:
            error(f"Error {type(e)} in LLM stream producer: {str(e)}")
        finally:
            remaining = collector.flush()
            for sentence in remaining:
                await text_queue.put(sentence)
            await text_queue.put(SENTINEL)

    producer_task = asyncio.create_task(producer())
    pending_item: Optional[str] = None
    text_queue_stop = False

    try:
        while True:
            if text_queue_stop:
                break

            if pending_item:
                item = pending_item
                pending_item = None
            else:
                item = await text_queue.get()
                if item == SENTINEL:
                    break

            current_batch: List[str] = [item]
            chars_cnt = len(item)
            text_queue.task_done()

            while not text_queue.empty():
                # todo: what if new item in context_size big?
                next_item = None
                try:
                    next_item = text_queue.get_nowait()
                    if next_item == SENTINEL:
                        text_queue_stop = True
                        next_item = None

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

                audio_iterator = stream_audio(http_session, tts_model, a_post_clone).__aiter__()

                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            audio_iterator.__anext__(),
                            timeout=10.
                        )
                        yield chunk

                    except StopAsyncIteration:
                        break
            except Exception as e:
                error(f"Error generating audio for batch '{full_text[:30]}...': {str(e)}")

    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass


async def encode_synthesized_stream(
        tts_model: ModelTTSAny,
        synthesizer: AsyncGenerator[str | bytes, None],
        output_format: Literal["pcm", "wav", "mp3", "ogg"],
) -> AsyncGenerator[str | bytes, None]:
    audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    result_queue: asyncio.Queue[Optional[str | bytes]] = asyncio.Queue()

    async def audio_source():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def input_processor():
        try:
            async for item in synthesizer:
                if isinstance(item, str):
                    await result_queue.put(item)
                else:
                    await audio_queue.put(item)
        except Exception as e:
            error(f"Stream input error: {e}")
        finally:
            await audio_queue.put(None)

    async def encoder_runner():
        try:
            async for chunk in encode_audio_stream(
                input_stream=audio_source(),
                output_format=output_format,
                sample_rate=tts_model.record.constants.sample_rate,
                channels=tts_model.record.constants.channels
            ):
                await result_queue.put(chunk)
        except Exception as e:
            error(f"Stream encoder error: {e}")
        finally:
            await result_queue.put(None)

    t_input = asyncio.create_task(input_processor())
    t_encoder = asyncio.create_task(encoder_runner())

    try:
        while True:
            res = await result_queue.get()
            if res is None:
                break
            yield res
    finally:
        t_input.cancel()
        t_encoder.cancel()
