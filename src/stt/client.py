import asyncio
from typing import AsyncGenerator, Tuple

import betterproto
from grpclib import GRPCError
from grpclib.client import Channel

from core.logger import error
from generated.stt_service import (
    ProtoTranscribeStub,
    TranscribePost,
    TranscribeStreamingConfig,
    PingRequest,

    SpeechStart, SpeechStop, SpeechTranscription
)
from stt.globals import GRPC_PORT


async def ping_stt(host: str) -> Tuple[bool, str | None]:
    async with Channel(host, GRPC_PORT) as channel:
        stub = ProtoTranscribeStub(channel)

        try:
            response = await asyncio.wait_for(
                stub.ping(PingRequest()),
                timeout=1.0
            )
            return response.status == "ok", None

        except Exception as e:
            return False, f"ping failed: {str(e)}"
        

async def stream_transcriptions(
        host: str,
        model: str,
        bytes_stream: AsyncGenerator[bytes, None],
) -> AsyncGenerator[SpeechStart | SpeechStop | SpeechTranscription, None]:
    async def generate_requests() -> AsyncGenerator[TranscribePost, None]:
        config_msg = TranscribeStreamingConfig(model=model)
        yield TranscribePost(config=config_msg)

        async for chunk in bytes_stream:
            yield TranscribePost(audio=chunk)

    async with Channel(host, GRPC_PORT) as channel:
        stub = ProtoTranscribeStub(channel)

        response_stream = stub.transcribe(generate_requests(), timeout=None)

        try:
            async for resp in response_stream:
                field, value = betterproto.which_one_of(resp, "output_event")
                if field == "speech_start":
                    assert value is not None
                    yield value

                elif field == "speech_stop":
                    assert value is not None
                    yield value

                elif field == "speech_transcription":
                    assert value is not None
                    yield value

                else:
                    continue

        except GRPCError as e:
            err = f"failed to stream_transcriptions: {str(e)}"
            error(err)
            raise e
