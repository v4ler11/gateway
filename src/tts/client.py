import asyncio
from typing import AsyncGenerator, Tuple

from grpclib import GRPCError
from grpclib.client import Channel

from core.logger import error
from generated.tts_audio import ProtoAudioStub, PingRequest
from tts.globals import GRPC_PORT
from tts.inference.schemas import TTSAudioPost


async def ping_tts(host: str) -> Tuple[bool, str | None]:
    async with Channel(host, GRPC_PORT) as channel:
        stub = ProtoAudioStub(channel)

        try:
            response = await asyncio.wait_for(
                stub.ping(PingRequest()),
                timeout=1.0
            )
            return response.status == "ok", None

        except Exception as e:
            return False, f"ping failed: {str(e)}"


async def stream_audio(
        host: str,
        post: TTSAudioPost
) -> AsyncGenerator[bytes, None]:
    async with Channel(host, GRPC_PORT) as channel:
        stub = ProtoAudioStub(channel)

        try:
            async for audio in stub.stream_audio(post.into_proto()):
                yield audio.data

        except GRPCError as e:
            err = f"failed to stream_audio_proto: {str(e)}"
            error(err)
            raise e
