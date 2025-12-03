from typing import Self, List, AsyncIterator

import aiohttp
from pydantic import BaseModel

from api.client import URLs
from core.routers.oai.models import ChatCompletionsResponseStreaming
from core.routers.oai.schemas import ChatPost
from core.routers.router_models import ModelsResponse, ModelResponse
from core.routers.utils import parse_sse_streaming


class AClient(BaseModel):
    urls: URLs
    session: aiohttp.ClientSession

    @classmethod
    def new(
            cls,
            base_url: str,
            session: aiohttp.ClientSession
    ) -> Self:
        return cls(
            urls=URLs(base_url=base_url),
            session=session
        )

    async def models(self) -> List[ModelResponse]:
        async with self.session.get(self.urls.models) as response:
            text = await response.text()

            if response.status != 200:
                raise Exception(f"Failed get models: {text}")

            models = ModelsResponse.model_validate_json(text)
            return models.data

    async def chat_completions(self, post: ChatPost) -> AsyncIterator[ChatCompletionsResponseStreaming]:
        if post.stream:
            async with self.session.post(
                url=self.urls.chat_completions,
                json=post.model_dump(),
            ) as response:
                async for chunk in parse_sse_streaming(response.content):
                    if chunk:
                        yield ChatCompletionsResponseStreaming.model_validate(chunk)

        else:
            raise NotImplementedError("Non-Streaming is not supported yet")
