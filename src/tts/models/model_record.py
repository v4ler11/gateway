from functools import partial
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from tts.models.urls import URLsKokoro


class ModelRecordBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    resolve_name: str

    @property
    def context_size(self) -> int:
        raise NotImplementedError("context_size must be implemented in ModelRecordBase")


class ParamsKokoro(BaseModel):
    voice: str
    speed: float


class ModelRecordKokoro(ModelRecordBase):
    files: List[str]
    params: ParamsKokoro
    urls: URLsKokoro | partial = Field(default=partial(URLsKokoro))

    @property
    def context_size(self) -> int:
        return 384
