from typing import Literal, Optional, Union, List, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from core.routers.oai.models import ChatMessage
from llm.models import SamplingParams
from tts.inference.utils import MEDIA_TYPES


class AudioPost(BaseModel):
    model: str
    text: str
    voice: str
    speed: float = Field(gt=0.0, le=5.0, default=1.0)
    response_format: Literal['pcm', 'wav', 'mp3', 'ogg']
    stream: bool = True

    @classmethod
    @field_validator("text")
    def validate_text(cls, v):
        if v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v

    def media_type(self):
        return MEDIA_TYPES[self.response_format]


class ChatTemplatesKwargs(BaseModel):
    reasoning_effort: Literal["low", "medium", "high"] = "low"


class ChatPostAudio(BaseModel):
    voice: Optional[str] = None
    format: Literal["pcm", "wav", "mp3", "ogg"]


class ChatPost(BaseModel):
    model: str
    messages: List[ChatMessage]
    modalities: List[Literal["audio", "text"]] = Field(default=["text"])
    audio: Optional[ChatPostAudio] = None
    stream: bool

    chat_template_kwargs: ChatTemplatesKwargs = Field(default_factory=ChatTemplatesKwargs)

    # sampling params
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    min_p: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    @classmethod
    @field_validator('modalities')
    def validate_modalities(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Modalities list cannot be empty')
        if len(v) != len(set(v)):
            raise ValueError('Modalities must be unique')
        return v

    @model_validator(mode='after')
    def validate_audio_requirements(self) -> Self:
        if "audio" in self.modalities and self.audio is None:
            raise ValueError("Field 'audio' is required when 'modalities' contains 'audio'.")
        if "audio" in self.modalities and not self.stream:
            raise ValueError("Field 'stream' must be True when 'modalities' contains 'audio' due to latency constraints.")
        return self

    def consume_sampling_params(self, sampling_params: SamplingParams) -> None:
        if sampling_params.max_tokens is not None:
            if self.max_tokens is not None:
                self.max_tokens = min(sampling_params.max_tokens, self.max_tokens)
            else:
                self.max_tokens = sampling_params.max_tokens
        if sampling_params.temperature is not None and self.temperature is None:
            self.temperature = sampling_params.temperature
        if sampling_params.min_p is not None and self.min_p is None:
            self.min_p = sampling_params.min_p
        if sampling_params.top_p is not None and self.top_p is None:
            self.top_p = sampling_params.top_p
        if sampling_params.top_k is not None and self.top_k is None:
            self.top_k = sampling_params.top_k
