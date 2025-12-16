from pathlib import Path
from typing import List, AsyncGenerator

from fastapi import UploadFile, File, Form
from starlette.responses import StreamingResponse

from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from models.definitions import ModelSTTAny
from stt.client import stream_transcriptions_proto
from stt.inference.ffmpeg import FfmpegDecoder


SUPPORTED_EXTENSIONS = ["wav", "mp3", "ogg", "flac", "opus"]


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


class OAIAudioTranscriptions(BaseRouter):
    def __init__(
            self,
            models: List[ModelSTTAny],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.models = models

        self.add_api_route("/oai/v1/audio/transcriptions", self._transcriptions, methods=["POST"])

    async def _transcriptions(
            self,
            file: UploadFile = File(...),
            model: str = Form(...),
    ):
        async def streamer() -> AsyncGenerator[str, None]:
            assert a_model is not None
            assert file is not None

            pcm_stream = get_pcm_stream(
                file_to_stream(file)
            )

            async for chunk in stream_transcriptions_proto(a_model, pcm_stream):
                yield chunk

        a_model = next((m for m in self.models if m.record.resolve_name == model), None)
        if not a_model:
            return error_constructor(
                message=f"Model {model} not found",
                error_type="model_not_found",
                status_code=404
            )

        if not file.filename:
            return error_constructor(
                message="file's filename is empty",
                error_type="invalid_file",
                status_code=422
            )

        ext = Path(file.filename).suffix.lstrip(".")
        if ext not in SUPPORTED_EXTENSIONS:
            return error_constructor(
                message=f"Unsupported file type {ext}. Only {', '.join(SUPPORTED_EXTENSIONS)} are supported",
                error_type="invalid_file",
                status_code=422
            )

        try:
            return StreamingResponse(streamer(), media_type="text/plain")

        except Exception as e:
            return error_constructor(
                message=f"Internal processing error: {str(e)}",
                error_type="internal_error",
                status_code=500
            )
