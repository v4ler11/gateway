from typing import Generator, Any, Literal

import torch

from kokoro import KPipeline
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import StreamingResponse, Response

from core.globals import MODELS_DIR
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from hf.download import download_repo_paths
from models.definitions import ModelTTSAny
from tts.inference.encode_audio_stream import encode_audio_stream
from tts.inference.utils import MEDIA_TYPES
from tts.models import ModelRecordKokoro


class AudioPost(BaseModel):
    model: str # todo: add validation when >1 model
    text: str
    voice: str
    speed: float = Field(gt=0.0, le=5.0, default=1.0)
    response_format: Literal["pcm", "wav", "mp3", "ogg"]
    stream: bool = True

    @classmethod
    @field_validator("text")
    def validate_text(cls, v):
        if v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v

    def media_type(self):
        return MEDIA_TYPES[self.response_format]


class AudioRouter(BaseRouter):
    def __init__(
            self,
            model: ModelTTSAny,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model

        self._pipeline: KPipeline

        self._init_pipeline(model)

        self.add_api_route(f"/v1/audio/stream", self._audio, methods=["POST"])


    def _init_pipeline(self, model: ModelTTSAny):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        d_paths, err = download_repo_paths(
            model.record.model,
            MODELS_DIR / model.record.model.replace("/", "_"),
            model.record.files,
        )
        if err is not None:
            raise RuntimeError(f"Failed to download model paths: {err}")

        pipeline = KPipeline(lang_code='a', device=device)

        for voice in model.record.voices:
            voice_pt = d_paths["voices"] / f"{voice}.pt"
            if not voice_pt.exists():
                raise RuntimeError(f"Failed to find voice file: {voice_pt}")
            voice_tensor = torch.load(str(voice_pt), weights_only=True).to(device)
            pipeline.voices[voice] = voice_tensor
        self._pipeline = pipeline


    async def _audio(self, post: AudioPost):
        assert isinstance(self.model.record, ModelRecordKokoro) # todo: remove when >1 model

        async def audio_streamer():
            # should be valid Float32
            stream: Generator[Any, None, None] = self._pipeline(
                text=post.text,
                voice=post.voice,
                speed=post.speed,
                split_pattern=None,
            )
            async def async_generator():
                async for _, _, a_tensor in iterate_in_threadpool(stream):
                    yield a_tensor.numpy().tobytes()

            async for chunk_ in encode_audio_stream(
                async_generator(),
                output_format=post.response_format,
                sample_rate=self.model.record.constants.sample_rate,
                channels=self.model.record.constants.channels,
            ):
                yield chunk_

        if len(post.text) > self.model.record.context_size:
            return error_constructor(
                message=f"Text is too long: {len(post.text)}; Max length is {self.model.record.context_size}",
                error_type="validation_error",
                status_code=400,
            )

        streamer = audio_streamer()
        if post.stream:
            return StreamingResponse(streamer, media_type=post.media_type())

        chunks = []
        async for chunk in audio_streamer():
            chunks.append(chunk)
        content = b"".join(chunks)

        return Response(content=content, media_type=post.media_type())
