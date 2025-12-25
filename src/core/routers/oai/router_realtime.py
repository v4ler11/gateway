import asyncio
from typing import List, AsyncGenerator, Tuple

import aiohttp
import pysbd

from fastapi import WebSocket, WebSocketDisconnect, status

from core.logger import error, info
from core.pipelines.chat_synthesized import stream_with_chat_synthesised
from core.routers.oai.models import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant, ChatMessage
from core.routers.oai.schemas import ChatPost
from core.routers.oai.utils import limit_messages, try_resolve_models, ResolvedModels
from core.routers.router_base import BaseRouter
from generated.stt_service import SpeechTranscription
from llm.client import stream_with_chat
from llm.models.prompts import LLM_TTS_PROMPT
from stt.client import stream_transcriptions
from stt.inference.ffmpeg_utils import get_pcm_stream
from models.definitions import ModelAny
from tts.inference.schemas import TTSAudioPost


BYTES_PER_SECOND = int(24000 * 1 * 4 * 1.3)
AUDIO_CHUNK_SIZE = 65_536 + 32_768


def prepare_post_bases(r_models: ResolvedModels) -> Tuple[ChatPost, TTSAudioPost]:
    assert r_models.llm is not None and r_models.tts is not None

    llm_post_base = ChatPost(
        model=r_models.llm.record.model,
        messages=[],
        stream=True
    )
    llm_post_base.consume_sampling_params(r_models.llm.sampling_params)

    tts_post = TTSAudioPost(
        model=r_models.tts.record.model,
        text="pass",
        voice=r_models.tts.record.params.voice,
        speed=r_models.tts.record.params.speed,
    )
    return llm_post_base, tts_post


def chunk_bytes(data: bytes, chunk_size: int) -> List[bytes]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


class OAIRealtimeRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelAny],
            http_session: aiohttp.ClientSession,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.http_session = http_session
        self.models = models

        self.add_api_websocket_route(
            "/oai/v1/realtime",
            self.realtime
        )

    async def realtime(self, websocket: WebSocket):
        await websocket.accept()

        model_name = websocket.query_params.get("model")
        if model_name is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing model parameter")
            return

        r_models_mb = try_resolve_models(model_name, self.models)
        if isinstance(r_models_mb, str):
            err = f"Failed to resolve models: {r_models_mb}"
            await websocket.send_json({"error": err})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=err)
            return

        r_models: ResolvedModels = r_models_mb
        if r_models_mb.llm is None or r_models.tts is None or r_models.stt is None:
            err = "Failed to resolve models: At least one of each: LLM, TTS, or STT models must be specified"
            await websocket.send_json({"error": err})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=err)
            return

        user_input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        audio_output_queue: asyncio.Queue[Tuple[bytes, int] | None] = asyncio.Queue()

        interrupt_event = asyncio.Event()
        current_turn_id = 0

        async def websocket_stream_adapter() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in websocket.iter_bytes():
                    yield chunk
            except WebSocketDisconnect:
                await user_input_queue.put(None)
                raise

        async def pace_audio(chunk_len: int):
            if chunk_len > 0:
                duration = chunk_len / BYTES_PER_SECOND
                await asyncio.sleep(duration)

        async def run_stt_producer():
            nonlocal current_turn_id
            assert r_models.stt is not None

            try:
                async for stt_resp in stream_transcriptions(
                        r_models.stt.config.container,
                        r_models.stt.record.model,
                        get_pcm_stream(websocket_stream_adapter())
                ):
                    if not isinstance(stt_resp, SpeechTranscription):
                        continue

                    current_turn_id += 1
                    interrupt_event.set()

                    info(f"USER SAYS: {stt_resp.text}")
                    user_input_queue.put_nowait(stt_resp.text)

            except Exception as e:
                error(f"STT Producer error: {e}")
                await user_input_queue.put(None)

        async def run_llm_tts_producer():
            assert r_models.llm is not None and r_models.tts is not None

            llm_post_base, tts_post = prepare_post_bases(r_models)
            messages: List[ChatMessage] = [
                ChatMessageSystem(content=LLM_TTS_PROMPT),
            ]

            while True:
                user_text = await user_input_queue.get()
                if user_text is None:
                    await audio_output_queue.put(None)
                    break

                interrupt_event.clear()
                processing_turn_id = current_turn_id

                messages.append(ChatMessageUser(content=user_text))
                messages = limit_messages(messages, r_models.llm)

                llm_post = llm_post_base.model_copy(update={"messages": messages})
                llm_stream = stream_with_chat(self.http_session, r_models.llm, llm_post)

                full_response_text = ""
                interrupted = False

                try:
                    async for chunk in stream_with_chat_synthesised(
                            r_models.tts,
                            tts_post,
                            llm_stream,
                            self.segmenter
                    ):
                        if interrupt_event.is_set() or processing_turn_id != current_turn_id:
                            interrupted = True
                            info("[USER INTERRUPTS ASSISTANT]")
                            break

                        if isinstance(chunk, bytes):
                            small_chunks = chunk_bytes(chunk, AUDIO_CHUNK_SIZE)
                            for small_chunk in small_chunks:
                                if interrupt_event.is_set():
                                    break
                                await audio_output_queue.put((small_chunk, processing_turn_id))

                        if isinstance(chunk, str):
                            full_response_text += chunk

                except Exception as e:
                    error(f"Error in LLM/TTS generation loop: {e}")

                content = full_response_text if not interrupted else f"{full_response_text} ... [user interrupted assistant here]"
                messages.append(ChatMessageAssistant(content=content))

        async def run_ws_sender():
            try:
                while True:
                    item = await audio_output_queue.get()
                    if item is None:
                        break

                    audio_chunk, turn_id = item

                    if turn_id != current_turn_id:
                        continue

                    await websocket.send_bytes(audio_chunk)
                    info("SENT AUDIO CHUNK. PACING...")
                    await pace_audio(len(audio_chunk))
                    info("PACING RELEASED")

            except Exception as e:
                error(f"Error in WS Sender: {e}")

        stt_task = asyncio.create_task(run_stt_producer())
        llm_tts_task = asyncio.create_task(run_llm_tts_producer())
        ws_sender_task = asyncio.create_task(run_ws_sender())

        try:
            await asyncio.wait(
                [stt_task, llm_tts_task, ws_sender_task],
                return_when=asyncio.FIRST_COMPLETED
            )
        except WebSocketDisconnect:
            pass

        stt_task.cancel()
        llm_tts_task.cancel()
        ws_sender_task.cancel()

        for task in [stt_task, llm_tts_task, ws_sender_task]:
            try:
                await task
            except asyncio.CancelledError:
                pass