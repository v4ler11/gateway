from typing import Generator, Any

import torch

from kokoro import KPipeline
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import StreamingResponse

from core.globals import MODELS_DIR
from core.logger import error
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from hf.download import download_repo_paths
from models.definitions import ModelTTSAny
from tts.inference.schemas import TTSAudioPost
from tts.models import ModelRecordKokoro


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


    async def _audio(self, post: TTSAudioPost):
        assert isinstance(self.model.record, ModelRecordKokoro) # todo: remove when >1 model

        async def audio_streamer():
            # should be valid Float32
            stream: Generator[Any, None, None] = self._pipeline(
                text=post.text,
                voice=post.voice,
                speed=post.speed,
                split_pattern=None,
            )
            async for _, _, a_tensor in iterate_in_threadpool(stream):
                yield a_tensor.numpy().tobytes()

        if len(post.text) > self.model.record.context_size:
            err = f"Text is too long: {len(post.text)}; Max length is {self.model.record.context_size}"
            error(err)
            return error_constructor(
                message=err,
                error_type="validation_error",
                status_code=400,
            )

        streamer = audio_streamer()
        return StreamingResponse(streamer, media_type="application/octet-stream")
