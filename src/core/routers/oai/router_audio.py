from typing import List, AsyncGenerator

import pysbd

from core.routers.oai.schemas import AudioPost
from core.routers.oai.sentence_collector import SentenceCollector
from core.routers.router_base import BaseRouter
from core.routers.schemas import error_constructor
from models.definitions import ModelTTSAny
from starlette.responses import StreamingResponse, Response

from tts.client import stream_audio
from tts.inference.encode_audio_stream import encode_audio_stream
from tts.inference.schemas import TTSAudioPost


def chunkify_text(
        text: str,
        model,
        segmenter: pysbd.Segmenter,
) -> List[str]:
    sentence_collector = SentenceCollector(segmenter=segmenter)
    sentences = sentence_collector.put(text)
    sentences.extend(sentence_collector.flush())

    batches = []
    current_batch_sentences = []
    current_char_count = 0

    limit = model.record.context_size * 0.9

    for s in sentences:
        s_len = len(s)

        if current_char_count + s_len + 1 > limit:
            if current_batch_sentences:
                batches.append(" ".join(current_batch_sentences))

            current_batch_sentences = [s]
            current_char_count = s_len
        else:
            current_batch_sentences.append(s)
            current_char_count += s_len + 1

    if current_batch_sentences:
        batches.append(" ".join(current_batch_sentences))

    return batches


class OAIAudioRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelTTSAny],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.models = models
        self.add_api_route("/oai/v1/audio/speech", self._generate_speech, methods=["POST"])

    async def _generate_speech(self, post: AudioPost):
        async def streamer() -> AsyncGenerator[bytes, None]:
            assert isinstance(batches, list)
            assert isinstance(model, ModelTTSAny)

            for batch in batches:
                a_post = TTSAudioPost(
                    model=model.record.model,
                    text=batch,
                    voice=post.voice,
                    speed=post.speed
                )
                async for audio_ in stream_audio(model.config.container, a_post):
                    yield audio_

        async def streamer_encoded(stream: AsyncGenerator[bytes, None]):
            assert isinstance(model, ModelTTSAny)

            async for audio_ in encode_audio_stream(
                input_stream=stream,
                output_format=post.response_format,
                sample_rate=model.record.constants.sample_rate,
                channels=model.record.constants.channels,
            ):
                yield audio_

        model = next((m for m in self.models if m.record.resolve_name == post.model), None)
        if not model:
            return error_constructor(
                message=f"Model {post.model} not found",
                error_type="model_not_found",
                status_code=404
            )

        try:
            batches = chunkify_text(post.text, model, self.segmenter)

            gen_encoded = streamer_encoded(streamer())

            if post.stream:
                return StreamingResponse(
                    gen_encoded,
                    media_type=post.media_type(),
                )

            content = b""
            async for audio in gen_encoded:
                content += audio

            return Response(content=content, media_type=post.media_type())


        except Exception as e:
            return error_constructor(
                message=f"Internal processing error: {str(e)}",
                error_type="internal_error",
                status_code=500
            )
