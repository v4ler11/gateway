import asyncio

from typing import List, AsyncGenerator, Counter

import aiohttp

from fastapi.responses import StreamingResponse

from core.logger import exception, info
from core.routers.oai.models import ChatPost, ChatCompletionsResponseStreaming, ChatMessage, \
    ChatCompletionsResponseNotStreaming
from core.routers.oai.utils import limit_messages
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from core.routers.utils import parse_sse_streaming
from models.s3_models.models import ModelAny


__all__ = ["OAIChatCompletionsRouter", "stream_with_chat"]


async def stream_with_chat(
        http_session: aiohttp.ClientSession,
        model: ModelAny,
        post: ChatPost,
) -> AsyncGenerator[ChatCompletionsResponseStreaming, None]:
    if not post.stream:
        raise ValueError(f"post.stream should be True, got post.stream={post.stream}")

    async with http_session.post(
        url=model.urls.chat_completions,
        json=post.model_dump(),
    ) as response:
        async for chunk in parse_sse_streaming(response.content):
            if chunk:
                yield ChatCompletionsResponseStreaming.model_validate(chunk)


def validate_messages(messages: List[ChatMessage]):
    role_counts = Counter(message.role for message in messages)
    if role_counts["system"] > 1:
        raise ValueError(f"Only one system role is allowed in messages, got {role_counts}:\n{messages}")


# reference : https://platform.openai.com/docs/api-reference/chat_streaming/streaming
class OAIChatCompletionsRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelAny],
            http_session: aiohttp.ClientSession,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.models = models
        self.http_session = http_session
        self.add_api_route(f"/oai/v1/chat/completions", self._chat_completions, methods=["POST"])

    async def _chat_completions(self, post: ChatPost):
        async def chat_completions_streamer() -> AsyncGenerator[str, None]:
            assert model is not None

            post.consume_sampling_params(model.sampling_params)

            if post.stream:
                async for chunk in stream_with_chat(
                    self.http_session,
                    model, post
                ):
                    chunk.model = model.record.resolve_name
                    yield chunk.to_streaming()

            else:
                async with self.http_session.post(
                    url=model.urls.chat_completions,
                    json=post.model_dump(),
                ) as response:
                    raw_comp = await response.json()
                    comp = ChatCompletionsResponseNotStreaming.model_validate(raw_comp)
                    comp.model = model.record.resolve_name
                    yield comp.model_dump_json()

        try:
            model = next((m for m in self.models if m.record.resolve_name == post.model), None)
            if not model:
                return error_constructor(
                    message=f"Model {post.model} not found",
                    error_type="model_not_found",
                    status_code=404
                )

            if not model.status.running:
                return error_constructor(
                    message=f"Model {post.model} is not running",
                    error_type="model_not_running",
                    status_code=400
                )

            info(f"Model name resolve {post.model} -> {model.record.model}")
            post.model = model.record.model

            messages = limit_messages(post.messages, model)
            validate_messages(messages)

            post.messages = messages

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
