from typing import AsyncGenerator

import aiohttp

from core.routers.oai.models import ChatCompletionsResponseStreaming
from core.routers.oai.schemas import ChatPost
from core.routers.utils import parse_sse_streaming
from models.definitions import ModelLLMAny


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
