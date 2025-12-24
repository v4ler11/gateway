from typing import AsyncIterator, Generator, Any

from kokoro import KPipeline
from starlette.concurrency import iterate_in_threadpool

from tts.inference.schemas import TTSAudioPost


async def stream_kokoro(pipeline: KPipeline, post: TTSAudioPost) -> AsyncIterator[bytes]:
    stream: Generator[Any, None, None] = pipeline(
        text=post.text,
        voice=post.voice,
        speed=post.speed,
        split_pattern=None,
    )
    async for _, _, a_tensor in iterate_in_threadpool(stream):
        yield a_tensor.numpy().tobytes()
