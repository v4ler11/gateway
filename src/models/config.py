import yaml

from typing import List, Dict, Any

from pydantic import BaseModel, ValidationError

from core.globals import YAML_CONFIG
from models.definitions import ModelConfigAny, MODEL_CONFIG_CLASSES, ModelConfigLLMAny, ModelConfigTTSAny, ModelAny

from llm.models.models import try_resolve_record as try_resolve_record_llm
from tts.models.models import try_resolve_record as try_resolve_record_tts


def validate_model_from_config(data: Dict[str, Any]) -> ModelConfigAny:
    validation_errors = []

    for model_class in MODEL_CONFIG_CLASSES:
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
    def read_yaml(cls) -> "Config":
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


def models_from_config(config: Config) -> List[ModelAny]:
    records = []

    for model in config.models:
        if isinstance(model, ModelConfigLLMAny):
            record = try_resolve_record_llm(model)
            records.append(record)

        elif isinstance(model, ModelConfigTTSAny):
            record = try_resolve_record_tts(model)
            records.append(record)

        else:
            raise ValueError(f"Unknown model type {type(model)}")

    return records
