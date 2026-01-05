import secrets
from dataclasses import dataclass
from typing import Iterable, Any, Dict, List, Optional

from core.logger import info
from core.routers.oai.models import ChatMessage, ChatMessageSystem
from models.definitions import ModelLLMAny, ModelTTSAny, ModelSTTAny, ModelAny


@dataclass
class ResolvedModels:
    llm: Optional[ModelLLMAny]
    tts: Optional[ModelTTSAny]
    stt: Optional[ModelSTTAny]


def generate_chat_completion_id(prefix: str) -> str:
    unique_suffix = secrets.token_urlsafe(9)[:12]  # 9 bytes -> ~12 chars when base64 encoded
    return f"{prefix}-{unique_suffix}"


def count_tokens(s: str) -> int:
    return len(s) // 4


def limit_messages(
        messages: List[ChatMessage],
        model: ModelAny
) -> List[ChatMessage]:
    new_messages = []

    messages_tok_limit: int = int(model.record.context_size * 0.95)

    take_messages = [
        isinstance(m, ChatMessageSystem) for m in messages
    ]
    tok_count = sum([
        count_tokens(m.content) for (m, take) in zip(messages, take_messages) if take
    ])

    take_messages.reverse()
    messages.reverse()

    for (message, take) in zip(messages, take_messages):
        if take:
            new_messages.append(message)
            continue

        m_tokens = count_tokens(message.content)
        if tok_count + m_tokens > messages_tok_limit:
            break

        tok_count += m_tokens
        new_messages.append(message)

    new_messages.reverse()

    info(f"model={model.record.resolve_name}; {tok_count=}; {messages_tok_limit=}")

    return new_messages


def convert_messages_to_chat_format(
        messages: Iterable[ChatMessage]
) -> List[Dict[str, Any]]:
    results = []
    for m in messages:
        try:
            results.append(m.to_chat_format())
        except AttributeError:
            raise NotImplementedError(f"Message of type `{type(m)}` does not have conversion to chat format")

    return results


def try_resolve_models(model_name: str, models: List[Any]) -> ResolvedModels | str:
    requested_names = [m.strip() for m in model_name.split("+") if m.strip()]
    available_models = {m.record.resolve_name: m for m in models}

    resolved_models = {"llm": [], "tts": [], "stt": []}

    for name in requested_names:
        if not (model := available_models.get(name)):
            return f"Model {name} is not available"

        if not model.status.running:
            return f"Model {name} is not running"

        info(f"Model name resolve {name} -> {model.record.model}")

        if isinstance(model, ModelLLMAny):
            resolved_models["llm"].append(model)
        elif isinstance(model, ModelTTSAny):
            resolved_models["tts"].append(model)
        elif isinstance(model, ModelSTTAny):
            resolved_models["stt"].append(model)
        else:
            raise ValueError(f"Unknown model type {type(model)}")

    if len(resolved_models["llm"]) > 1 or len(resolved_models["tts"]) > 1 or len(resolved_models["stt"]) > 1:
        return f"Only one of each: LLM, TTS, or STT models can be specified"

    return ResolvedModels(
        llm=resolved_models["llm"][0] if resolved_models["llm"] else None,
        tts=resolved_models["tts"][0] if resolved_models["tts"] else None,
        stt=resolved_models["stt"][0] if resolved_models["stt"] else None,
    )
