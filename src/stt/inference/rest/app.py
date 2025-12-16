import asyncio
from typing import List, Any

from fastapi import FastAPI

from core.routers.router_base import BaseRouter
from models.definitions import ModelSTTAny
from stt.inference.rest.router_transcriptions import TranscriptionsRouter


class App(FastAPI):
    def __init__(
            self,
            models: List[ModelSTTAny],
            p_model: Any,
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loop: asyncio.AbstractEventLoop
        self.models = models
        self.p_model = p_model

        self.add_event_handler("startup", self._startup_events)
        self.add_event_handler("shutdown", self._shutdown_events)

    @classmethod
    def new(
            cls,
            models: List[ModelSTTAny],
            p_model: Any,
    ) -> "App":
        return cls(
            models=models,
            p_model=p_model,

            docs_url=None,
            redoc_url=None,
            openapi_url="/v1/openapi.json",
            openapi_tags=[],
        )

    async def _startup_events(self):
        self.loop = asyncio.get_event_loop()

        for router in self._routers():
            self.include_router(router)

    async def _shutdown_events(self):
        pass

    def _routers(self):
        return [
            BaseRouter(),
            TranscriptionsRouter(
                loop=self.loop,
                model=self.models[0],
                p_model=self.p_model,
            ),
        ]
