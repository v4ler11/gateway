from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, Field, model_validator

from models.s1_records import EngineParamsAny, SamplingParams
from models.s1_records.engine_params import EngineParamsLlamacpp


class ModelConfig(BaseModel):
    model: str
    sampling_params: Optional[SamplingParams] = None

    @property
    def base_url(self) -> str:
        raise NotImplementedError(f"url property not implemented ModelConfig")


class ModelConfigLocal(ModelConfig):
    container: str
    port: int = Field(ge=1, le=65535)
    engine_params: Optional[EngineParamsAny] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.container}:{self.port}"


class ModelLocalBackend(Enum):
    llamacpp = "llamacpp"


class ModelConfigLocalLlamaCpp(ModelConfigLocal):
    backend: ModelLocalBackend

    @model_validator(mode='after')
    def validate_backend_compatibility(self):
        if self.backend == ModelLocalBackend.llamacpp and self.engine_params is not None:
            if not isinstance(self.engine_params, EngineParamsLlamacpp):
                raise ValueError("engine_params must be of type EngineParamsLlamacpp when backend is llamacpp")

        return self


class ModelConfigRemote(ModelConfig):
    url: str

    @property
    def base_url(self) -> str:
        return self.url

    @classmethod
    @field_validator("url")
    def validate_url(cls, v):
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("URL must start with either http:// or https://")
        return v


class ModelConfigRemoteLmStudio(ModelConfigRemote):
    backend: Literal["lmstudio"]
