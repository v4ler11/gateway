import asyncio
import json
from typing import Any, AsyncGenerator

from pydantic import BaseModel
from starlette.responses import StreamingResponse

from core.routers.router_base import BaseRouter
from models.definitions import ModelSTTAny
from stt.inference.stream_utils import transcribe_bytes
from stt.inference.streaming_parakeet import StreamingParakeet
from stt.models import ModelRecordParakeet


class TransPost(BaseModel):
    model: str
    audio: bytes


class TranscriptionsRouter(BaseRouter):
    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            p_model: Any,
            model: ModelSTTAny,
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loop = loop
        self.p_model = p_model
        self.model = model
        assert isinstance(self.model.record, ModelRecordParakeet) # todo: remove when >1 model

        self.add_api_route(f"/v1/audio/transcriptions", self._transcriptions, methods=["POST"])

    async def _transcriptions(self, post: TransPost):
        async def streamer() -> AsyncGenerator[str, None]:
            p_streamer = StreamingParakeet(self.p_model)

            async for word_stamp, is_final in transcribe_bytes(
                self.loop, post.audio, p_streamer
            ):
                yield json.dumps(dict(
                    text=word_stamp.word,
                    start=word_stamp.start,
                    end=word_stamp.end,
                    is_final=is_final
                ))

        return StreamingResponse(streamer(), media_type="application/json")
