from typing import List

import aiohttp
from fastapi.responses import Response, StreamingResponse

from core.routers.oai.schemas import AudioPost
from core.routers.oai.stream_utils import stream_audio
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from models.definitions import ModelTTSAny
from models.urls import URLs


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
        post.model = model.record.model

        assert isinstance(model.record.urls, URLs)
        target_url = model.record.urls.generate

        try:
            if post.stream:
                return StreamingResponse(
                    stream_audio(self.http_session, model, post),
                    media_type=post.media_type(),
                )

            async with self.http_session.post(target_url, json=post.model_dump()) as response:
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
