from dataclasses import field
from typing import List, Union, Literal, Optional, Dict, Any

from pydantic import BaseModel, model_serializer, Field

from core.routers.utils import str_to_streaming
from models.s1_records import SamplingParams

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


class ChatDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None

    @model_serializer # role should be absent in dict if none
    def serialize_model(self) -> Dict[str, Any]:
        data = {
            'content': self.content,
            'reasoning_content': self.reasoning_content
        }
        if self.role:
            data['role'] = self.role
        return data


class ChatCompletionsResponseChoiceStreaming(_ChatCompletionsResponseChoice):
    delta: ChatDelta | Dict


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
    created: int
    model: str
    timings: Optional[ChatTimings] = None
    service_tier: Optional[Any] = None
    system_fingerprint: Optional[Any] = None
    prompt_logprobs: Optional[Any] = None
    kv_transfer_params: Optional[Any] = None


class ChatCompletionsResponseNotStreaming(_ChatCompletionsResponse):
    object: Literal["chat.completion"]
    choices: List[ChatCompletionsResponseChoiceNonStreaming]


class ChatCompletionsResponseStreaming(_ChatCompletionsResponse):
    object: Literal["chat.completion.chunk"]
    choices: List[ChatCompletionsResponseChoiceStreaming]

    def to_streaming(self) -> str:
        return str_to_streaming(self.model_dump_json())


class ChatTemplatesKwargs(BaseModel):
    reasoning_effort: Literal["low", "medium", "high"] = "low"


class ChatPost(BaseModel):
    model: str
    messages: List[ChatMessage]
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
