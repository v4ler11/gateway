from functools import partial
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from stt.models.urls import URLsSTT


class ModelRecordBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    resolve_name: str
    caps: List[str] = ["stt"]

    urls: URLsSTT | partial = Field(default=partial(URLsSTT))

    @property
    def context_size(self) -> int:
        raise NotImplementedError("context_size must be implemented in ModelRecordBase")


class ModelRecordParakeet(ModelRecordBase):
    files: List[str]
