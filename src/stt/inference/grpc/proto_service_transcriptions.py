import asyncio
from typing import AsyncIterator, Any

import numpy as np
from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError

from core.logger import warn, exception, info
from generated.stt_service import (
    ProtoTranscribeBase,

    TranscribeResp, TranscribePost,
    TranscribeStreamingConfig,

    PingResponse, PingRequest
)

from models.definitions import ModelSTTAny
from stt.inference.schemas import SpeechStop, SpeechStart, SpeechTranscription

from stt.inference.streaming_parakeet import stream_parakeet_with_vad
from stt.models import ModelRecordParakeet


class ProtoTranscriptionService(ProtoTranscribeBase):
    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            parakeet_model: Any,
            vad_model: Any,
            model: ModelSTTAny,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loop = loop

        self.parakeet_model = parakeet_model
        self.vad_model = vad_model

        self.model = model
        assert isinstance(self.model.record, ModelRecordParakeet) # todo: remove when >1 model

    async def transcribe(
            self,
            transcribe_post_iterator: AsyncIterator[TranscribePost]
    ) -> AsyncIterator[TranscribeResp]:

        try:
            first_resp = await transcribe_post_iterator.__anext__()
        except StopAsyncIteration:
            return

        config = first_resp.config
        if not isinstance(config, TranscribeStreamingConfig):
            err = f"First chunk must be a streaming config, got {config}"
            warn(err)
            raise GRPCError(Status.FAILED_PRECONDITION, err)

        async def bytes_generator() -> AsyncIterator[np.ndarray]:
            SAMPLE_RATE = 16000
            BYTES_PER_SAMPLE = 4
            CHUNK_DURATION = 0.2
            TARGET_SIZE = int(SAMPLE_RATE * CHUNK_DURATION * BYTES_PER_SAMPLE)

            buffer = bytearray()

            try:
                async for chunk in transcribe_post_iterator:
                    if chunk.audio:
                        buffer.extend(chunk.audio)

                        while len(buffer) >= TARGET_SIZE:
                            chunk_bytes = buffer[:TARGET_SIZE]
                            del buffer[:TARGET_SIZE]
                            yield np.frombuffer(chunk_bytes, dtype=np.float32)

                if len(buffer) > 0:
                    yield np.frombuffer(buffer, dtype=np.float32)

            except (asyncio.CancelledError, StreamTerminatedError):
                return

        streamer = stream_parakeet_with_vad(
            loop=self.loop,
            audio_stream=bytes_generator(),
            model=self.parakeet_model,
            vad_model=self.vad_model,
        )

        try:
            async for event in streamer:
                response = None

                if isinstance(event, SpeechStart):
                    response = TranscribeResp(speech_start=event.to_proto())
                elif isinstance(event, SpeechStop):
                    response = TranscribeResp(speech_stop=event.to_proto())
                elif isinstance(event, SpeechTranscription):
                    response = TranscribeResp(speech_transcription=event.to_proto())

                if response:
                    try:
                        print(response)
                        yield response
                    except (StreamTerminatedError, asyncio.CancelledError):
                        info("Client disconnected during response stream.")
                        return

        except asyncio.CancelledError:
            info("Stream handler cancelled.")
            raise

        except Exception as e:
            err = f"Unexpected error in transcribe: {str(e)}"
            exception(err)
            raise GRPCError(Status.UNKNOWN, err)

    async def ping(self, ping_request: PingRequest) -> PingResponse:
        return PingResponse(status="ok")
