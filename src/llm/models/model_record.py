from functools import partial
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, ConfigDict

from llm.models.engine_params import EngineParamsLlamacpp
from llm.models.urls import URLsLlamaCpp, URLsLmStudio
from models.utils import validate_huggingface_path


class SamplingParams(BaseModel):
    max_tokens: int = Field(gt=0, default=4096)
    temperature: float = Field(ge=0.0, default=0.6)
    top_p: Optional[float] = Field(gt=0.0, le=1.0, default=None)
    top_k: Optional[float] = Field(ge=0.0, default=None)
    min_p: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    presence_penalty: Optional[float] = Field(ge=-2.0, le=2.0, default=None)


class ModelRecordBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    resolve_name: str
    caps: List[str] = ["text-generation"]

    tokenizer: str
    sampling_params: SamplingParams

    @classmethod
    @field_validator('tokenizer')
    def validate_tokenizer(cls, v):
        return validate_huggingface_path(v)

    @classmethod
    @field_validator('resolve_name')
    def validate_resolve_name(cls, v):
        if '/' in v or ' ' in v:
            raise ValueError("resolve_name cannot contain '/' or spaces")
        return v

    @property
    def context_size(self):
        raise NotImplementedError("context_size is not implemented for ModelRecordBase")


class ModelRecordLlamaCpp(ModelRecordBase):
    model_file: str
    engine_params: EngineParamsLlamacpp
    urls: URLsLlamaCpp | partial = Field(default=partial(URLsLlamaCpp)) # create constructor now, initialize later

    @property
    def context_size(self):
        return self.engine_params.context_size


class ModelRecordLMStudio(ModelRecordBase):
    ctx_size: int = Field(ge=1)
    urls: URLsLmStudio | partial = Field(default=partial(URLsLmStudio))

    @property
    def context_size(self):
        return self.ctx_size
