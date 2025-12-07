from typing import AsyncGenerator, Literal

from tts.inference.ffmpeg import FfmpegProc, FfmpegParams
from tts.inference.utils import build_wav_header


async def encode_audio_stream(
        input_stream: AsyncGenerator[bytes, None],
        output_format: Literal["pcm", "wav", "mp3", "ogg"],
        sample_rate: int,
        channels: int
) -> AsyncGenerator[bytes, None]:
    if output_format == "pcm":
        async for chunk in input_stream:
            yield chunk
        return

    if output_format == "wav":
        yield build_wav_header(sample_rate, channels)
        async for chunk in input_stream:
            yield chunk
        return

    async with FfmpegProc(
            input_stream=input_stream,
            params=FfmpegParams(
                output_format=output_format,
                sample_rate=sample_rate,
                channels=channels
            ),
    ) as stream:
        async for chunk in stream:
            yield chunk
