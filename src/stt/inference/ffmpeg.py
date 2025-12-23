import asyncio

from typing import AsyncGenerator, Optional


class FfmpegDecoder:
    def __init__(
            self,
            input_stream: AsyncGenerator[bytes, None],
    ):
        self.input_stream = input_stream
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._feeder_task: Optional[asyncio.Task] = None

    async def _feed_stdin(self):
        assert self._proc is not None
        assert self._proc.stdin is not None

        try:
            async for chunk in self.input_stream:
                self._proc.stdin.write(chunk)
                await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                    await self._proc.stdin.wait_closed()
                except (RuntimeError, BrokenPipeError, ConnectionResetError):
                    pass

    async def __aenter__(self):
        cmd = [
            "ffmpeg",
            "-i", "pipe:0",
            "-f", "f32le",
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            "pipe:1"
        ]

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        if not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError("Failed to open ffmpeg pipes")

        self._feeder_task = asyncio.create_task(self._feed_stdin())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._feeder_task:
            if not self._feeder_task.done():
                self._feeder_task.cancel()
            try:
                await self._feeder_task
            except asyncio.CancelledError:
                pass

        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._proc or not self._proc.stdout:
            raise StopAsyncIteration

        if self._feeder_task and self._feeder_task.done():
            try:
                self._feeder_task.result()
            except Exception as e:
                raise RuntimeError(f"Input stream failed: {e}") from e

        # Read 4096 bytes (2048 samples @ 16-bit) -> approx 128ms of audio
        data = await self._proc.stdout.read(4096)

        if data:
            return data

        return_code = await self._proc.wait()

        if return_code != 0:
            stderr_out = await self._proc.stderr.read() if self._proc.stderr else b""
            err_msg = stderr_out.decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg failed (code {return_code}): {err_msg}")

        raise StopAsyncIteration
