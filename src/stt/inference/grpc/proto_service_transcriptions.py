import asyncio
from typing import AsyncIterator, Any

from grpclib import GRPCError, Status

from core.logger import warn, exception
from generated.stt_service import ProtoTranscribeBase, TranscribeResp, TranscribePost, TranscribeStreamingConfig

from models.definitions import ModelSTTAny
from stt.inference.stream_utils import transcribe_byte_stream
from stt.inference.streaming_parakeet import StreamingParakeet
from stt.models import ModelRecordParakeet


class ProtoTranscriptionService(ProtoTranscribeBase):
    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            p_model: Any,
            model: ModelSTTAny,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loop = loop
        self.p_model = p_model
        self.model = model
        assert isinstance(self.model.record, ModelRecordParakeet) # todo: remove when >1 model

    async def transcribe(
            self,
            transcribe_post_iterator: AsyncIterator[TranscribePost]
    ) -> AsyncIterator[TranscribeResp]:
        first_resp = await transcribe_post_iterator.__anext__()
        config = first_resp.config
        if not isinstance(config, TranscribeStreamingConfig):
            err = f"First chunk must be a streaming config, got {config}"
            warn(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

        async def bytes_generator() -> AsyncIterator[bytes]:
            async for chunk in transcribe_post_iterator:
                if chunk.audio is None:
                    continue

                yield chunk.audio

        streamer = StreamingParakeet(self.p_model)
        try:
            async for word_stamp, is_final in transcribe_byte_stream(
                self.loop, bytes_generator(), streamer
            ):
                yield TranscribeResp(
                    text=word_stamp.word,
                    start=word_stamp.start,
                    end=word_stamp.end,
                    is_final=is_final
                )
        except Exception as e:
            err = f"failed to transcribe: {str(e)}"
            exception(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)
