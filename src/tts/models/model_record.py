from functools import partial
from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tts.models.urls import URLsKokoro


class ModelRecordBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    resolve_name: str
    caps: List[str] = ["tts"]

    @property
    def context_size(self) -> int:
        raise NotImplementedError("context_size must be implemented in ModelRecordBase")


class ParamsKokoro(BaseModel):
    voice: str
    speed: float


class ConstantsKokoro(BaseModel):
    sample_rate: int = 24000
    channels: int = 1
    bytes_per_sample: int = 4


class ModelRecordKokoro(ModelRecordBase):
    files: List[str]
    voices: List[str]
    params: ParamsKokoro
    urls: URLsKokoro | partial = Field(default=partial(URLsKokoro))
    constants: ConstantsKokoro = Field(default_factory=ConstantsKokoro)

    @property
    def context_size(self) -> int:
        return 384

    @model_validator(mode='after')
    def validate_voice_is_available(self) -> 'ModelRecordKokoro':
        current_voice = self.params.voice

        if current_voice not in self.voices:
            raise ValueError(
                f"Configured voice '{current_voice}' is not in the list "
                f"of available voices: {self.voices}"
            )

        return self
