from typing import Optional, Any, Literal

from pydantic import BaseModel


class ModelConfigBase(BaseModel):
    model: str
    backend: Any

    container: str

    params: Optional[Any] = None


class ModelConfigParakeet(ModelConfigBase):
    backend: Literal["parakeet"]
