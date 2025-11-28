from typing import Optional, Any, Literal

from pydantic import BaseModel, Field

from tts.models.model_record import ParamsKokoro


class ModelConfigBase(BaseModel):
    model: str
    backend: Any

    container: str
    port: int = Field(ge=1, le=65535)

    params: Optional[Any] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.container}:{self.port}"


class ModelConfigKokoro(ModelConfigBase):
    backend: Literal["kokoro"]
    params: Optional[ParamsKokoro] = None
