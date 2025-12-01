import asyncio
import struct

from typing import AsyncGenerator, Literal


def build_wav_header(sample_rate: int, channels: int) -> bytes:
    bytes_per_sample = 4
    audio_format = 3

    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    bits_per_sample = 32

    return (
            b'RIFF' +
            b'\xff\xff\xff\xff' +
            b'WAVEfmt ' +
            b'\x10\x00\x00\x00' +
            struct.pack('<H', audio_format) +
            struct.pack('<H', channels) +
            struct.pack('<I', sample_rate) +
            struct.pack('<I', byte_rate) +
            struct.pack('<H', block_align) +
            struct.pack('<H', bits_per_sample) +
            b'data' +
            b'\xff\xff\xff\xff'
    )


async def encode_audio_stream(
        input_stream: AsyncGenerator[bytes, None],
        output_format: Literal["pcm", "wav", "mp3", "ogg"],
        sample_rate: int = 24000,
        channels: int = 1
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

    if output_format == "mp3":
        fmt_args = ["-f", "mp3", "-b:a", "128k"]
    elif output_format == "ogg":
        fmt_args = ["-f", "ogg", "-c:a", "libopus", "-b:a", "32k"]
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    cmd = [
              "ffmpeg",
              "-f", "f32le",
              "-ar", str(sample_rate),
              "-ac", str(channels),
              "-i", "pipe:0"
          ] + fmt_args + ["pipe:1"]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    stdin_stream = proc.stdin
    stdout_stream = proc.stdout

    if stdin_stream is None or stdout_stream is None:
        raise RuntimeError("Failed to open ffmpeg pipes")

    async def feed_stdin():
        try:
            async for chunk in input_stream:
                stdin_stream.write(chunk)
                await stdin_stream.drain()
            stdin_stream.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    feeder_task = asyncio.create_task(feed_stdin())

    try:
        while True:
            data = await stdout_stream.read(4096)
            if not data:
                break
            yield data
    finally:
        feeder_task.cancel()
        try:
            await feeder_task
        except asyncio.CancelledError:
            pass

        if proc.returncode is None:
            try:
                proc.terminate()
                await proc.wait()
            except ProcessLookupError:
                pass
