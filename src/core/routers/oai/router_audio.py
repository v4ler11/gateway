from typing import Literal, List

import aiohttp
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from models.definitions import ModelTTSAny
from models.urls import URLs
from tts.inference.utils import MEDIA_TYPES


class AudioPost(BaseModel):
    model: str
    text: str
    voice: str
    speed: float = Field(gt=0.0, le=5.0, default=1.0)
    response_format: Literal['pcm', 'wav', 'mp3', 'ogg']
    stream: bool = True

    @classmethod
    @field_validator("text")
    def validate_text(cls, v):
        if v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v

    def media_type(self):
        return MEDIA_TYPES[self.response_format]


def payload_from_post(model: ModelTTSAny, post: AudioPost):
    return {
        "model": model.record.model,
        "text": post.text,
        "voice": post.voice,
        "speed": post.speed,
        "response_format": post.response_format,
        "stream": post.stream
    }


class OAIAudioRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelTTSAny],
            http_session: aiohttp.ClientSession,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.models = models
        self.http_session = http_session
        self.add_api_route("/oai/v1/audio/speech", self._generate_speech, methods=["POST"])

    async def _generate_speech(self, post: AudioPost):
        model = next((m for m in self.models if m.record.resolve_name == post.model), None)
        if not model:
            return error_constructor(
                message=f"Model {post.model} not found",
                error_type="model_not_found",
                status_code=404
            )

        payload = payload_from_post(model, post)
        assert isinstance(model.record.urls, URLs)
        target_url = model.record.urls.generate

        try:
            if post.stream:
                async def proxy_stream():
                    try:
                        async with self.http_session.post(target_url, json=payload) as resp_:
                            if resp_.status != 200:
                                error_text = await resp_.text()
                                yield error_text.encode()
                                return

                            async for chunk in resp_.content.iter_any():
                                yield chunk
                    except Exception as e: # noqa
                        pass

                return StreamingResponse(
                    proxy_stream(),
                    media_type=post.media_type(),
                )

            async with self.http_session.post(target_url, json=payload) as response:
                if response.status != 200:
                    return error_constructor(
                        message=f"Inference error: {await response.text()}",
                        error_type="inference_error",
                        status_code=response.status
                    )
                content = await response.read()

                filename = f"speech.{post.response_format}"
                return Response(
                    content=content,
                    media_type=post.media_type(),
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )

        except aiohttp.ClientError as e:
            return error_constructor(
                message=f"Failed to connect to inference service: {str(e)}",
                error_type="connection_error",
                status_code=503
            )
        except Exception as e:
            return error_constructor(
                message=f"Internal processing error: {str(e)}",
                error_type="internal_error",
                status_code=500
            )
