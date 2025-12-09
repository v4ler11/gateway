from kokoro import KPipeline
from starlette.responses import StreamingResponse

from core.logger import error
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from models.definitions import ModelTTSAny
from tts.inference.schemas import TTSAudioPost
from tts.inference.stream_utils import audio_streamer
from tts.models import ModelRecordKokoro


class AudioRouter(BaseRouter):
    def __init__(
            self,
            model: ModelTTSAny,
            pipeline: KPipeline,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        assert isinstance(self.model.record, ModelRecordKokoro) # todo: remove when >1 model

        self._pipeline = pipeline

        self.add_api_route(f"/v1/audio/stream", self._audio, methods=["POST"])


    async def _audio(self, post: TTSAudioPost):
        if len(post.text) > self.model.record.context_size:
            err = f"Text is too long: {len(post.text)}; Max length is {self.model.record.context_size}"
            error(err)
            return error_constructor(
                message=err,
                error_type="validation_error",
                status_code=400,
            )

        streamer = audio_streamer(self._pipeline, post)
        return StreamingResponse(streamer, media_type="application/octet-stream")
