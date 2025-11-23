import yaml

from typing import Literal, List, Dict, Any

from pydantic import BaseModel, field_validator, ValidationError

from core.globals import YAML_CONFIG

type ModelAny = ModelExternalLmStudio


class Model(BaseModel):
    pass


class ModelExternal(Model):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("URL must start with either http:// or https://")
        return v


class ModelExternalLmStudio(ModelExternal):
    kind: Literal["lmstudio"]


def validate_model_from_config(data: Dict[str, Any]) -> ModelAny:
    model_classes = [ModelExternalLmStudio]

    validation_errors = []

    for model_class in model_classes:
        try:
            return model_class.model_validate(data)
        except ValidationError as e:
            validation_errors.append(f"{model_class.__name__}: {str(e)}")
            continue

    error_msg = "Failed to validate data with any model type:\n" + "\n".join(validation_errors)
    raise ValueError(error_msg)


class Config(BaseModel):
    models: List[ModelAny]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if "models" not in data:
            raise ValueError(f"Config must contain models key, got {data}")

        models = data["models"]

        if not isinstance(models, list):
            raise ValueError("models must be a list")

        for idx, m in enumerate(models):
            if not isinstance(m, dict):
                raise ValueError(f"model {idx=} must be a dictionary, got {type(m)}")

        return cls(
            models=[
                validate_model_from_config(model_data)
                for model_data in models
            ]
        )


def read_config() -> Config:
    yaml_data = yaml.safe_load(YAML_CONFIG.read_text())
    config = Config.from_dict(yaml_data)
    return config
