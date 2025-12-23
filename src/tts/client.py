from typing import AsyncGenerator

import aiohttp
from grpclib import GRPCError
from grpclib.client import Channel

from core.logger import error
from generated.tts_audio import ProtoAudioStreamStub
from models.definitions import ModelTTSAny
from models.urls import URLs
from tts.inference.schemas import TTSAudioPost


async def stream_audio(
        http_session: aiohttp.ClientSession,
        model: ModelTTSAny,
        post: TTSAudioPost
) -> AsyncGenerator[bytes, None]:
    assert isinstance(model.record.urls, URLs)

    try:
        async with http_session.post(
                url=model.record.urls.generate,
                json=post.model_dump()
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                yield error_text.encode()
                return

            async for chunk in response.content.iter_any():
                if chunk:
                    yield chunk

    except Exception as e:  # noqa
        error(f"failed to stream_audio: {str(e)}")
        pass


async def stream_audio_proto(
        model: ModelTTSAny,
        post: TTSAudioPost
):
    async with Channel(model.config.container, 50051) as channel: # todo: do not hardcode here
        stub = ProtoAudioStreamStub(channel)

        try:
            async for audio in stub.stream_audio(post.into_proto()):
                yield audio.data

        except GRPCError as e:
            err = f"failed to stream_audio_proto: {str(e)}"
            error(err)
            raise e
