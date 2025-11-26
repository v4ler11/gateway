import secrets
from typing import Iterable, Any, Dict, List

from core.logger import info
from core.routers.oai.models import ChatMessage, ChatMessageSystem
from models.s3_models.models import ModelAny


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
