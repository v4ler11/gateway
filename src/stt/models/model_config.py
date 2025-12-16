from typing import Optional, Any, Literal

from pydantic import BaseModel, Field


class ModelConfigBase(BaseModel):
    model: str
    backend: Any

    container: str
    port: int = Field(ge=1, le=65535)

    params: Optional[Any] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.container}:{self.port}"


class ModelConfigParakeet(ModelConfigBase):
    backend: Literal["Parakeet"]
