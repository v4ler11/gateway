import asyncio

from typing import AsyncIterator, AsyncGenerator, Tuple

import numpy as np

from stt.inference.streaming_parakeet import StreamingParakeet, WordStamp


async def transcribe_bytes(
        loop: asyncio.AbstractEventLoop,
        data: bytes,
        streamer: StreamingParakeet
) -> AsyncGenerator[Tuple[WordStamp, bool], None]:
    audio_full = np.frombuffer(data, dtype=np.float32)

    stamps = await loop.run_in_executor(
        None,
        streamer.transcribe_chunk,
        audio_full
    )

    for stamp in stamps:
        yield stamp, False

    final_stamps = await loop.run_in_executor(None, streamer.finish)
    for stamp in final_stamps:
        yield stamp, True


async def transcribe_byte_stream(
        loop: asyncio.AbstractEventLoop,
        byte_stream_iterator: AsyncIterator[bytes],
        streamer: StreamingParakeet
) -> AsyncGenerator[Tuple[WordStamp, bool], None]:
    async for chunk_bytes in byte_stream_iterator:
        audio_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)

        stamps = await loop.run_in_executor(
            None,
            streamer.transcribe_chunk,
            audio_chunk
        )

        for stamp in stamps:
            yield stamp, False

    final_stamps = await loop.run_in_executor(None, streamer.finish)
    for stamp in final_stamps:
        yield stamp, True
