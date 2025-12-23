from typing import List

import aiohttp

from fastapi import FastAPI

from core.routers.oai.router_audio import OAIAudioRouter
from core.routers.oai.router_chat_completions import OAIChatCompletionsRouter
from core.routers.oai.router_models import OAIModelsRouter
from core.routers.oai.router_transcriptions import OAIAudioTranscriptions
from core.routers.router_base import BaseRouter
from core.routers.router_models import ModelsRouter
from models.definitions import ModelAny, ModelLLMAny, ModelSTTAny, ModelTTSAny


__all__ = ["App"]


class App(FastAPI):
    def __init__(
            self,
            models: List[ModelAny],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.http_session: aiohttp.ClientSession
        self.models = models
        self.add_event_handler("startup", self._startup_events)
        self.add_event_handler("shutdown", self._shutdown_events)

    @classmethod
    def new(cls, models: List[ModelAny]) -> "App":
        return cls(
            models=models,
            docs_url=None,
            redoc_url=None,
            openapi_url="/v1/openapi.json",
            openapi_tags=[],
        )

    async def _startup_events(self):
        self.http_session = aiohttp.ClientSession()

        for router in self._routers():
            self.include_router(router)

    async def _shutdown_events(self):
        await self.http_session.close()

    def _routers(self):
        return [
            BaseRouter(),
            ModelsRouter(models=self.models),

            # OAI Routers
            OAIModelsRouter(
                models=[m for m in self.models if isinstance(m, ModelLLMAny)]
            ),
            OAIChatCompletionsRouter(
                models=self.models,
                http_session=self.http_session
            ),
            OAIAudioRouter(
                models=[m for m in self.models if isinstance(m, ModelTTSAny)],
            ),
            OAIAudioTranscriptions(
                models=[m for m in self.models if isinstance(m, ModelSTTAny)],
            )
        ]
