from typing import List

from tts.models import ModelRecordAny, ModelRecordKokoro

__all__ = ["RECORDS"]

from tts.models.model_record import ParamsKokoro


RECORDS: List[ModelRecordAny] = [
    ModelRecordKokoro(
        model="hexgrad/Kokoro-82M",
        resolve_name="kokoro",
        files=["kokoro-v1_0.pth", "voices"],
        params=ParamsKokoro(
            voice="af_heart",
            speed=1.
        )
    )
]
