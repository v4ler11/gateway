from typing import Optional, Any, Literal

from pydantic import BaseModel, Field

from tts.models.model_record import ParamsKokoro


class ModelConfigBase(BaseModel):
    model: str
    backend: Any
    container: str

    params: Optional[Any] = None


class ModelConfigKokoro(ModelConfigBase):
    backend: Literal["kokoro"]
    params: Optional[ParamsKokoro] = None
