from dataclasses import dataclass
from typing import List, AsyncGenerator

import aiohttp
import pysbd
from fastapi import WebSocket, WebSocketDisconnect, status

from core.logger import info
from core.pipelines.chat_synthesized import stream_with_chat_synthesised, encode_synthesized_stream
from core.routers.oai.models import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant, ChatMessage
from core.routers.oai.schemas import ChatPost
from core.routers.oai.utils import limit_messages
from core.routers.router_base import BaseRouter
from generated.stt_service import SpeechTranscription
from llm.client import stream_with_chat
from llm.models.prompts import LLM_TTS_PROMPT
from stt.client import stream_transcriptions
from stt.inference.ffmpeg_utils import get_pcm_stream
from models.definitions import ModelLLMAny, ModelTTSAny, ModelSTTAny, ModelAny
from tts.inference.schemas import TTSAudioPost


@dataclass
class ResolvedModels:
    llm: ModelLLMAny
    tts: ModelTTSAny
    stt: ModelSTTAny


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

    def _try_resolve_models(self, model_name: str) -> ResolvedModels | str:
        requested_names = [m.strip() for m in model_name.split("+") if m.strip()]
        available_models = {m.record.resolve_name: m for m in self.models}

        resolved_models = {"llm": [], "tts": [], "stt": []}

        for name in requested_names:
            if not (model := available_models.get(name)):
                return f"Model {name} is not available"

            if not model.status.running:
                return f"Model {name} is not running"

            info(f"Model name resolve {name} -> {model.record.model}")

            if isinstance(model, ModelLLMAny):
                resolved_models["llm"].append(model)
            elif isinstance(model, ModelTTSAny):
                resolved_models["tts"].append(model)
            elif isinstance(model, ModelSTTAny):
                resolved_models["stt"].append(model)
            else:
                raise ValueError(f"Unknown model type {type(model)}")

        if not resolved_models["llm"] or not resolved_models["tts"] or not resolved_models["stt"]:
            return f"At least one of each: LLM, TTS, or STT models must be specified"

        if len(resolved_models["llm"]) > 1 or len(resolved_models["tts"]) > 1 or len(resolved_models["stt"]) > 1:
            return f"Only one of each: LLM, TTS, or STT models can be specified"

        return ResolvedModels(
            llm=resolved_models["llm"][0],
            tts=resolved_models["tts"][0],
            stt=resolved_models["stt"][0],
        )

    async def realtime(self, websocket: WebSocket):
        async def websocket_stream_adapter() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in websocket.iter_bytes():
                    yield chunk
            except WebSocketDisconnect:
                raise
            except Exception as e:
                raise

        model_name = websocket.query_params.get("model")
        if model_name is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing model parameter")
            return

        r_models_mb = self._try_resolve_models(model_name)
        if isinstance(r_models_mb, str):
            err = f"Failed to resolve models: {r_models_mb}"
            await websocket.send_json({"error": err})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=err)
            return

        r_models: ResolvedModels = r_models_mb

        await websocket.accept()

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

        pcm_stream = get_pcm_stream(websocket_stream_adapter())
        messages: List[ChatMessage] = [
            ChatMessageSystem(content=LLM_TTS_PROMPT),
        ]

        async for stt_resp in stream_transcriptions(
            r_models.stt.config.container,
            r_models.stt.record.model,
            pcm_stream
        ):
            if not isinstance(stt_resp, SpeechTranscription):
                continue

            messages.append(ChatMessageUser(content=stt_resp.text))

            messages = limit_messages(messages, r_models.llm)

            llm_post = llm_post_base.model_copy(update={"messages": messages})

            llm_stream = stream_with_chat(
                self.http_session,
                r_models.llm, llm_post
            )

            synthesizer = stream_with_chat_synthesised(
                r_models.tts,
                tts_post,
                llm_stream,
                self.segmenter
            )

            encoded_synthesizer = encode_synthesized_stream(
                r_models.tts,
                synthesizer,
                "pcm",
            )

            text_response = ""
            async for chunk in encoded_synthesizer:
                if isinstance(chunk, bytes):
                    await websocket.send_bytes(chunk)

                if isinstance(chunk, str):
                    text_response += chunk

            messages.append(ChatMessageAssistant(content=text_response))
