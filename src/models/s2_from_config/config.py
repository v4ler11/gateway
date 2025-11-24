import yaml

from typing import List, Dict, Any

from pydantic import BaseModel, ValidationError

from core.globals import YAML_CONFIG
from models.s2_from_config import ModelConfigAny, MODEL_CLASSES


def validate_model_from_config(data: Dict[str, Any]) -> ModelConfigAny:
    validation_errors = []

    for model_class in MODEL_CLASSES:
        try:
            return model_class.model_validate(data)
        except ValidationError as e:
            validation_errors.append(f"{model_class.__name__}: {str(e)}")
            continue

    error_msg = "Failed to validate data with any model type:\n" + "\n".join(validation_errors)
    raise ValueError(error_msg)


class Config(BaseModel):
    models: List[ModelConfigAny]

    @classmethod
    def read(cls) -> "Config":
        yaml_data = yaml.safe_load(YAML_CONFIG.read_text())
        return Config.from_dict(yaml_data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
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
