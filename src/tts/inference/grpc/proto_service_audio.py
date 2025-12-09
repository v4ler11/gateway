from grpclib import GRPCError, Status
from kokoro import KPipeline

from core.logger import warn, error

from models.definitions import ModelTTSAny
from generated.tts_audio import ProtoAudioStreamBase, ProtoResp, ProtoPost
from tts.inference.schemas import TTSAudioPost
from tts.inference.stream_utils import audio_streamer
from tts.models import ModelRecordKokoro


class ProtoAudioService(ProtoAudioStreamBase):
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

    async def stream_audio(self, proto_post: ProtoPost):
        try:
            post = TTSAudioPost.from_proto(proto_post)
        except Exception as e:
            err = f"failed to validate post: {str(e)}"
            warn(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

        if len(post.text) > self.model.record.context_size:
            err = f"Text is too long: {len(post.text)}; Max length is {self.model.record.context_size}"
            warn(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

        try:
            async for audio in audio_streamer(self._pipeline, post):
                yield ProtoResp(data=audio)

        except Exception as e:
            err = f"failed to stream audio: {str(e)}"
            error(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)
