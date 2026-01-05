import asyncio

from typing import AsyncGenerator, List, Optional, Literal
from pydantic import BaseModel


class FfmpegParams(BaseModel):
    output_format: Literal["mp3", "ogg"]
    sample_rate: int
    channels: int


class FfmpegProc:
    def __init__(
            self,
            input_stream: AsyncGenerator[bytes, None],
            params: FfmpegParams,
    ):
        self.input_stream = input_stream
        self.params = params
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._feeder_task: Optional[asyncio.Task] = None

    def _get_format_args(self) -> List[str]:
        if self.params.output_format == "mp3":
            return ["-f", "mp3", "-b:a", "128k"]
        elif self.params.output_format == "ogg":
            return ["-f", "ogg", "-c:a", "libopus", "-b:a", "32k"]
        else:
            raise ValueError(f"Unsupported format: {self.params.output_format}")

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
            "-f", "f32le",
            "-ar", str(self.params.sample_rate),
            "-ac", str(self.params.channels),
            "-i", "pipe:0"
        ] + self._get_format_args() + ["pipe:1"]

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

        data = await self._proc.stdout.read(4096)

        if data:
            return data

        return_code = await self._proc.wait()

        if return_code != 0:
            stderr_out = await self._proc.stderr.read() if self._proc.stderr else b""
            raise RuntimeError(
                f"FFmpeg failed (code {return_code}): {stderr_out.decode('utf-8', errors='ignore')}"
            )

        raise StopAsyncIteration