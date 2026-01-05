from typing import List

from stt.models import ModelRecordAny, ModelRecordParakeet


RECORDS: List[ModelRecordAny] = [
    ModelRecordParakeet(
        model="nvidia/parakeet-tdt-0.6b-v3",
        resolve_name="parakeet",
        files=[],
    ),
]
