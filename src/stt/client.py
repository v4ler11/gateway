from typing import AsyncGenerator

import aiohttp
from grpclib import GRPCError
from grpclib.client import Channel

from core.logger import error
from generated.stt_service import ProtoTranscribeStub, TranscribePost, TranscribeStreamingConfig
from models.definitions import ModelSTTAny
from models.urls import URLs
from stt.inference.rest.router_transcriptions import TransPost


async def stream_transcriptions(
        http_session: aiohttp.ClientSession,
        model: ModelSTTAny,
        post: TransPost,
):
    assert isinstance(model.record.urls, URLs)

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


async def stream_transcriptions_proto(
        model: ModelSTTAny,
        bytes_stream: AsyncGenerator[bytes, None],
) -> AsyncGenerator[str, None]:
    async def generate_requests() -> AsyncGenerator[TranscribePost, None]:
        config_msg = TranscribeStreamingConfig(model=model.record.model)
        yield TranscribePost(config=config_msg)

        async for chunk in bytes_stream:
            yield TranscribePost(audio=chunk)

    async with Channel(model.config.container, model.config.port) as channel:
        stub = ProtoTranscribeStub(channel)
        response_stream = stub.transcribe(generate_requests())

        try:
            async for response in response_stream:
                if response.text:
                    yield response.text

        except GRPCError as e:
            err = f"failed to stream_transcriptions_proto: {str(e)}"
            error(err)
            raise e
