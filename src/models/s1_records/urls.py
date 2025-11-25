from pydantic import BaseModel, field_validator


class URLs(BaseModel):
    url: str

    @classmethod
    @field_validator("url")
    def validate_url(cls, v):
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("URL must start with either http:// or https://")

        if v.endswith("/"):
            raise ValueError("URL must not end with /")

        return v

    @property
    def ping(self) -> str:
        raise NotImplementedError("ping is not implemented for URLs")

    @property
    def chat_completions(self) -> str:
        raise NotImplementedError("chat_completions is not implemented for URLs")

    @property
    def models(self) -> str:
        raise NotImplementedError("models is not implemented for URLs")


class URLsLlamaCpp(URLs):
    @property
    def ping(self) -> str:
        return f"{self.url}/health"

    @property
    def chat_completions(self) -> str:
        return f"{self.url}/v1/chat/completions"

    @property
    def models(self) -> str:
        return f"{self.url}/v1/models"


class URLsLmStudio(URLs):
    pass
