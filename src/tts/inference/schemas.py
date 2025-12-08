from typing import Self

from pydantic import field_validator, BaseModel, Field

from generated.tts_audio import ProtoPost


class TTSAudioPost(BaseModel):
    model: str # todo: add validation when >1 model
    text: str
    voice: str
    speed: float = Field(gt=0.0, le=5.0, default=1.0)

    @classmethod
    def from_proto(cls, proto: ProtoPost) -> Self:
        return cls(
            model=proto.model,
            text=proto.text,
            voice=proto.voice,
            speed=proto.speed,
        )

    def into_proto(self) -> ProtoPost:
        return ProtoPost(
            model=self.model,
            text=self.text,
            voice=self.voice,
            speed=self.speed,
        )

    @classmethod
    @field_validator("text")
    def validate_text(cls, v):
        if v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v
