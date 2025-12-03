import base64
from typing import List

from pydantic import BaseModel

from core.routers.oai.models import ChatCompletionsResponseStreaming


class ChunksAudio(BaseModel):
    transcripts: List[str] = []
    audio: bytes = b""


def collect_chunks_streaming_audio(chunks: List[ChatCompletionsResponseStreaming]) -> ChunksAudio:
    res = ChunksAudio()

    for chunk in chunks:
        if choices := chunk.choices:
            ch0 = choices[0]
            delta = ch0.delta
            if audio := delta.audio if not isinstance(delta, dict) else delta.get("audio"):
                if transcript := audio.transcript:
                    res.transcripts.append(transcript)

                if data := audio.data:
                    data = base64.b64decode(data)
                    res.audio += data

    return res
