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
    def generate(self) -> str:
        raise NotImplementedError("generate is not implemented for URLs")
