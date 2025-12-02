from typing import List, Iterator

import requests

from pydantic import BaseModel, field_validator

from core.routers.oai.models import ChatCompletionsResponseStreaming
from core.routers.oai.schemas import ChatPost
from core.routers.router_models import ModelResponse, ModelsResponse


class URLs(BaseModel):
    base_url: str

    @classmethod
    @field_validator("url")
    def validate_url(cls, v):
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("URL must start with either http:// or https://")

        if v.endswith("/"):
            raise ValueError("URL must not end with /")

        return v

    @property
    def models(self) -> str:
        return f"{self.base_url}/v0/models"

    @property
    def chat_completions(self) -> str:
        return f"{self.base_url}/oai/v1/chat/completions"


class Client:
    def __init__(self, addr: str):
        self._urls = URLs(base_url=addr)
        self._session = requests.session()

    def models(self) -> List[ModelResponse]:
        with self._session.get(self._urls.models) as resp:
            resp.raise_for_status()
            resp = ModelsResponse.model_validate(resp.json())
            return resp.data

    def chat_completions(self, post: ChatPost) -> Iterator[ChatCompletionsResponseStreaming]:
        if post.stream:
            with self._session.post(
                self._urls.chat_completions,
                json=post.model_dump(),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        line = line.strip()
                        if line.startswith("data: "):
                            line = line[6:]
                        resp = ChatCompletionsResponseStreaming.model_validate_json(line)
                        yield resp

        else:
            raise NotImplementedError("Non-Streaming is not supported yet")
