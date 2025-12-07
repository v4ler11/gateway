from pydantic import field_validator, BaseModel, Field


class TTSAudioPost(BaseModel):
    model: str # todo: add validation when >1 model
    text: str
    voice: str
    speed: float = Field(gt=0.0, le=5.0, default=1.0)

    @classmethod
    @field_validator("text")
    def validate_text(cls, v):
        if v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v
