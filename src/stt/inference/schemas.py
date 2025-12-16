from pydantic import BaseModel


class STTPost(BaseModel):
    model: str
