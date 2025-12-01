import base64
import secrets
from dataclasses import field
from typing import List, Union, Literal, Optional, Dict, Any, Self

from pydantic import BaseModel, model_serializer, model_validator

from core.routers.utils import str_to_streaming


type ChatMessage = Union[ChatMessageSystem, ChatMessageUser, ChatMessageAssistant]


class ChatMessageBase(BaseModel):
    role: Any
    content: str

    def to_chat_format(self) -> Dict[str, Any]:
        return dict(
            role=self.role,
            content=self.content,
        )


class ChatMessageSystem(ChatMessageBase):
    role: Literal["system"] = "system"


class ChatMessageUser(ChatMessageBase):
    role: Literal["user"] = "user"


class ChatMessageAssistant(ChatMessageBase):
    role: Literal["assistant"] = "assistant"
    reasoning_content: Optional[str] = None

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        data = {
            "role": self.role,
        }
        if self.content is not None:
            data["content"] = self.content
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content
        return data


class _ChatCompletionsResponseChoice(BaseModel):
    index: Optional[int] = 0
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class ChatCompletionsResponseChoiceNonStreaming(_ChatCompletionsResponseChoice):
    message: ChatMessage
    refusal: Optional[Any] = None
    annotations: Optional[Any] = None
    audio: Optional[Any] = None
    function_call: Optional[Any] = None
    tool_calls: List[Any] = field(default_factory=list)
    reasoning_content: Optional[str] = None

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        data = {
            "index": self.index,
            'message': self.message,
            "finish_reason": self.finish_reason,
        }
        if self.logprobs is not None:
            data["logprobs"] = self.logprobs
        if self.refusal is not None:
            data["refusal"] = self.refusal
        if self.annotations is not None:
            data["annotations"] = self.annotations
        if self.audio is not None:
            data["audio"] = self.audio
        if self.function_call is not None:
            data["function_call"] = self.function_call
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content

        return data


class AudioResponse(BaseModel):
    id: Optional[str] = None # sent once in a first chunk
    data: Optional[str] = None # base64
    transcript: Optional[str] = None

    @classmethod
    def new(cls, any_data: str | bytes, first_chunk: bool) -> Self:
        id_ = None
        data = None
        transcript = None
        if isinstance(any_data, str):
            transcript = any_data
        if isinstance(any_data, bytes):
            data = base64.b64encode(any_data).decode("utf-8")
        if first_chunk:
            id_ = AudioResponse.generate_id()

        return cls(id=id_, data=data, transcript=transcript)

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        data = {}
        if self.id is not None:
            data['id'] = self.id
        if self.data is not None:
            data['data'] = self.data
        if self.transcript is not None:
            data['transcript'] = self.transcript
        return data

    @staticmethod
    def generate_id() -> str:
        return f"audio_{secrets.token_hex(12)}"

    @model_validator(mode='after')
    def validate_fields(self) -> Self:
        if not self.data and not self.transcript:
            raise ValueError("Either data or transcript should be provided")
        return self


class ChatDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    audio: Optional[AudioResponse] = None
    reasoning_content: Optional[str] = None

    @model_serializer # role should be absent in dict if none
    def serialize_model(self) -> Dict[str, Any]:
        data = {}
        if self.role is not None:
            data['role'] = self.role
        if self.content is not None:
            data['content'] = self.content
        if self.audio is not None:
            data['audio'] = self.audio.model_dump()
        if self.reasoning_content is not None:
            data['reasoning_content'] = self.reasoning_content
        return data


class ChatCompletionsResponseChoiceStreaming(_ChatCompletionsResponseChoice):
    delta: ChatDelta | Dict

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        data = {
            "index": self.index,
            "delta": self.delta,
            "finish_reason": self.finish_reason,
        }
        if self.logprobs is not None:
            data["logprobs"] = self.logprobs
        return data



class ChatUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int
    prompt_tokens_details: Optional[int] = None


class ChatTimings(BaseModel):
    cache_n: int
    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float


class _ChatCompletionsResponse(BaseModel):
    id: str

    object: Any
    choices: Any

    created: int
    model: str
    timings: Optional[ChatTimings] = None
    service_tier: Optional[Any] = None
    system_fingerprint: Optional[Any] = None
    prompt_logprobs: Optional[Any] = None
    kv_transfer_params: Optional[Any] = None

    @staticmethod
    def generate_id() -> str:
        return f"msg_{secrets.token_hex(12)}"

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "object": self.object,
            "choices": self.choices,
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
        }
        if self.timings is not None:
            data["timings"] = self.timings
        if self.service_tier is not None:
            data["service_tier"] = self.service_tier
        if self.prompt_logprobs is not None:
            data["prompt_logprobs"] = self.prompt_logprobs
        if self.kv_transfer_params is not None:
            data["kv_transfer_params"] = self.kv_transfer_params
        return data


class ChatCompletionsResponseNotStreaming(_ChatCompletionsResponse):
    object: Literal["chat.completion"] = "chat.completion"
    choices: List[ChatCompletionsResponseChoiceNonStreaming]


class ChatCompletionsResponseStreaming(_ChatCompletionsResponse):
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    choices: List[ChatCompletionsResponseChoiceStreaming]

    def to_streaming(self) -> str:
        return str_to_streaming(self.model_dump_json())
