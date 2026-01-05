from typing import List

from tts.models import ModelRecordAny, ModelRecordKokoro
from tts.models.model_record import ParamsKokoro


__all__ = ["RECORDS"]


RECORDS: List[ModelRecordAny] = [
    ModelRecordKokoro(
        model="hexgrad/Kokoro-82M",
        resolve_name="kokoro",
        files=["voices"], # todo: failed to load model form disk
        voices=[
            "af_heart",
        ],
        params=ParamsKokoro(
            voice="af_heart",
            speed=1.
        )
    )
]
