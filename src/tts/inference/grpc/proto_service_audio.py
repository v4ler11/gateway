from grpclib import GRPCError, Status
from kokoro import KPipeline

from core.logger import warn, error

from models.definitions import ModelTTSAny
from generated.tts_audio import ProtoAudioBase, AudioResp, AudioPost, PingRequest, PingResponse
from tts.inference.schemas import TTSAudioPost
from tts.inference.streaming_kokoro import stream_kokoro
from tts.models import ModelRecordKokoro


class ProtoAudioService(ProtoAudioBase):
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

    async def stream_audio(self, audio_post: AudioPost):
        try:
            post = TTSAudioPost.from_proto(audio_post)
        except Exception as e:
            err = f"failed to validate post: {str(e)}"
            warn(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

        if len(post.text) > self.model.record.context_size:
            err = f"Text is too long: {len(post.text)}; Max length is {self.model.record.context_size}"
            warn(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

        try:
            async for audio in stream_kokoro(self._pipeline, post):
                yield AudioResp(data=audio)

        except Exception as e:
            err = f"failed to stream audio: {str(e)}"
            error(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

    async def ping(self, ping_request: PingRequest) -> PingResponse:
        return PingResponse(status="ok")
