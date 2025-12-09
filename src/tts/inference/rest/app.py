from typing import List

from fastapi import FastAPI
from kokoro import KPipeline

from core.routers.router_base import BaseRouter
from models.definitions import ModelTTSAny
from tts.inference.rest.router_audio import AudioRouter


__all__ = ["App"]


class App(FastAPI):
    def __init__(
            self,
            models: List[ModelTTSAny],
            pipeline: KPipeline,
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.models = models
        self.pipeline = pipeline
        self.add_event_handler("startup", self._startup_events)
        self.add_event_handler("shutdown", self._shutdown_events)

    @classmethod
    def new(
            cls,
            models: List[ModelTTSAny],
            pipeline: KPipeline
    ) -> "App":
        return cls(
            models=models,
            pipeline=pipeline,
            docs_url=None,
            redoc_url=None,
            openapi_url="/v1/openapi.json",
            openapi_tags=[],
        )

    async def _startup_events(self):
        for router in self._routers():
            self.include_router(router)

    async def _shutdown_events(self):
        pass

    def _routers(self):
        return [
            BaseRouter(),
            AudioRouter(model=self.models[0], pipeline=self.pipeline),
        ]
