from typing import AsyncGenerator

from fastapi import UploadFile

from stt.inference.ffmpeg import FfmpegDecoder


async def file_to_stream(file: UploadFile, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
    await file.seek(0)
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        yield chunk


async def get_pcm_stream(file_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    async with FfmpegDecoder(
        input_stream=file_stream,
    ) as stream:
        async for chunk in stream:
            yield chunk
