import asyncio
import time

from typing import List, AsyncGenerator, Counter, Optional

import aiohttp
import pysbd

from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from core.logger import exception, info
from core.routers.oai.models import (
    ChatMessage, ChatCompletionsResponseNotStreaming, ChatCompletionsResponseStreaming,
    ChatCompletionsResponseChoiceStreaming, ChatDelta, AudioResponse, ChatMessageSystem
)
from core.routers.oai.schemas import ChatPost, AudioPost, ChatPostAudio
from core.routers.oai.stream_utils import stream_with_chat, stream_with_chat_synthesised, encode_synthesized_stream
from core.routers.oai.utils import limit_messages
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from llm.models.prompts import LLM_TTS_PROMPT
from models.definitions import ModelLLMAny, ModelTTSAny, ModelAny
from tts.inference.schemas import TTSAudioPost


__all__ = ["OAIChatCompletionsRouter"]


def validate_messages(messages: List[ChatMessage]):
    role_counts = Counter(message.role for message in messages)
    if role_counts["system"] > 1:
        raise ValueError(f"Only one system role is allowed in messages, got {role_counts}:\n{messages}")


def include_system_if_needed(post: ChatPost, model: ModelLLMAny) -> List[ChatMessage]:
    messages = list(post.messages)

    system_message = next((m for m in messages if isinstance(m, ChatMessageSystem)), None)

    needs_tts = "audio" in post.modalities

    if system_message:
        if needs_tts and LLM_TTS_PROMPT not in system_message.content:
            prefix = "\n\n" if system_message.content else ""
            system_message.content += f"{prefix}{LLM_TTS_PROMPT}"
    else:
        base_content = model.record.prompt if model.record.prompt else ""

        if needs_tts:
            separator = "\n\n" if base_content else ""
            final_content = f"{base_content}{separator}{LLM_TTS_PROMPT}"
        else:
            final_content = base_content

        if final_content:
            new_system_message = ChatMessageSystem(content=final_content)
            messages.insert(0, new_system_message)

    return messages


def llm_chat_post_from_post(
        post: ChatPost,
        model: ModelLLMAny,
        messages: List[ChatMessage],
) -> ChatPost:
    return ChatPost(
        model=model.record.model,
        messages=messages,
        stream=post.stream,
        chat_template_kwargs=post.chat_template_kwargs,
        max_tokens=post.max_tokens,
        temperature=post.temperature,
        min_p=post.min_p,
        top_p=post.top_p,
        top_k=post.top_k,
        stop=post.stop,

        repetition_penalty=post.repetition_penalty,
        presence_penalty=post.presence_penalty,
    )


class ResolvedModels(BaseModel):
    llm: ModelLLMAny
    tts: Optional[ModelTTSAny]


# reference : https://platform.openai.com/docs/api-reference/chat_streaming/streaming
class OAIChatCompletionsRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelAny],
            http_session: aiohttp.ClientSession,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.models = models
        self.http_session = http_session
        self.add_api_route(f"/oai/v1/chat/completions", self._chat_completions, methods=["POST"])

    def _try_resolve_post_models(self, post: ChatPost) -> ResolvedModels | Response:
        requested_names = [m.strip() for m in post.model.split("+") if m.strip()]
        available_models = {m.record.resolve_name: m for m in self.models}

        resolved_llms = []
        resolved_tts = []

        for name in requested_names:
            model = available_models.get(name)

            if not model:
                return error_constructor(
                    message=f"Model '{name}' not found",
                    error_type="model_not_found",
                    status_code=404
                )
            if not model.status.running:
                return error_constructor(
                    message=f"Model '{name}' is not running",
                    error_type="model_not_running",
                    status_code=400
                )

            info(f"Model name resolve {name} -> {model.record.model}")

            if isinstance(model, ModelLLMAny):
                resolved_llms.append(model)
            elif isinstance(model, ModelTTSAny):
                resolved_tts.append(model)
            else:
                return error_constructor(
                    message=f"Model type '{type(model)}' is not supported",
                    error_type="server_error",
                    status_code=500
                )

        if not resolved_llms:
            return error_constructor(
                message="LLM is required for chat/completions",
                error_type="validation_error",
                status_code=422
            )

        if len(resolved_llms) > 1:
            return error_constructor(
                message=f"Only one LLM is allowed, got: {[m.record.resolve_name for m in resolved_llms]}",
                error_type="validation_error",
                status_code=422
            )
        if len(resolved_tts) > 1:
            return error_constructor(
                message=f"Only one TTS model is allowed, got: {[m.record.resolve_name for m in resolved_tts]}",
                error_type="validation_error",
                status_code=422
            )

        if "audio" in post.modalities and not resolved_tts:
            return error_constructor(
                message="TTS model is required for chat/completions if 'audio' is in modalities",
                error_type="validation_error",
                status_code=422
            )

        return ResolvedModels(
            llm=resolved_llms[0],
            tts=resolved_tts[0] if resolved_tts else None
        )

    async def _chat_completions(self, post: ChatPost):
        async def chat_completions_streamer() -> AsyncGenerator[str, None]:
            assert r_models is not None

            chat_post.consume_sampling_params(r_models.llm.sampling_params)

            if post.stream:
                llm_stream = stream_with_chat(
                    self.http_session,
                    r_models.llm, chat_post
                )
                if r_models.tts is None:
                    async for chunk in llm_stream:
                        chunk.model = r_models.llm.record.resolve_name
                        yield chunk.to_streaming()

                else:
                    assert isinstance(r_models.tts, ModelTTSAny)
                    assert isinstance(post.audio, ChatPostAudio)

                    a_post = TTSAudioPost(
                        model=r_models.tts.record.model,
                        text="pass",
                        voice=post.audio.voice or r_models.tts.record.params.voice,
                        speed=r_models.tts.record.params.speed,
                    )

                    synthesizer = stream_with_chat_synthesised(
                        self.http_session,
                        r_models.tts,
                        a_post,
                        llm_stream,
                        self.segmenter
                    )
                    resp_chunk_base = ChatCompletionsResponseStreaming(
                        id=ChatCompletionsResponseStreaming.generate_id(),
                        choices=[],
                        created=int(time.time()),
                        model=post.model,
                    )
                    idx = 0
                    async for chunk in encode_synthesized_stream(
                        r_models.tts,
                        synthesizer,
                        post.audio.format
                    ):
                        if "text" not in post.modalities and isinstance(chunk, str):
                            continue

                        audio = AudioResponse.new(chunk, idx == 0)
                        idx += 1

                        choice = ChatCompletionsResponseChoiceStreaming(
                            delta=ChatDelta(
                                audio=audio,
                            )
                        )
                        resp_chunk_base_clone = resp_chunk_base.model_copy(update={"choices": [choice]})
                        yield resp_chunk_base_clone.to_streaming()

                    finish_choice = ChatCompletionsResponseChoiceStreaming(
                        delta={},
                        finish_reason="stop"
                    )
                    resp_chunk_base_clone = resp_chunk_base.model_copy(update={"choices": [finish_choice]})
                    yield resp_chunk_base_clone.to_streaming()

            else:
                if r_models.tts is None:
                    async with self.http_session.post(
                        url=r_models.llm.urls.generate,
                        json=post.model_dump(),
                    ) as response:
                        raw_comp = await response.json()
                        comp = ChatCompletionsResponseNotStreaming.model_validate(raw_comp)
                        comp.model = r_models.llm.record.resolve_name
                        yield comp.model_dump_json()
                else:
                    raise ValueError("Voice modality is only supported with stream=True due to latency constraints.")

        try:
            r_models_mb = self._try_resolve_post_models(post)
            if isinstance(r_models_mb, Response):
                return r_models_mb
            r_models = r_models_mb

            messages = include_system_if_needed(post, r_models.llm)

            messages = limit_messages(messages, r_models.llm)
            validate_messages(messages)

            chat_post = llm_chat_post_from_post(post, r_models.llm, messages)

            streamer = chat_completions_streamer()

            return StreamingResponse(streamer, media_type="text/event-stream")

        except asyncio.TimeoutError:
            return error_constructor(
                message=f"Request timeout",
                error_type="request_timeout",
                status_code=408
            )

        except Exception as e:
            exception(e)
            return error_constructor(
                message=f"Internal server error: {str(e)}",
                error_type="internal_server_error",
                status_code=500
            )
