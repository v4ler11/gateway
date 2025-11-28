from functools import partial
from typing import Optional, Any

from pydantic import BaseModel, Field, ConfigDict
from transformers import AutoTokenizer

from core.globals import TOKENIZERS_DIR
from hf.download import download_repo_files
from llm.models import (
    ModelRecordLocalAny, ModelRecordRemoteAny, URLsLocalAny,
    URLsRemoteAny, EngineParamsAny, SamplingParams, ModelRecordAny,
    ModelConfigLocalAny, ModelConfigRemoteAny, ModelConfigAny
)
from llm.models.records import RECORDS
from models.status import Status


class ModelBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Any
    status: Status = Field(default_factory=Status)


def try_get_tokenizer(record: ModelRecordAny) -> Any:
    paths, error = download_repo_files(
        record.tokenizer,
        TOKENIZERS_DIR / record.tokenizer.replace("/", "_"),
        [
            "tokenizer.json",
            "config.json",
        ]
    )
    if error is not None:
        raise ValueError(f"Failed to download tokenizer {record.tokenizer}: {error}")

    tokenizer_path = paths.get("tokenizer.json")
    assert tokenizer_path is not None

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path.parent))

    return tokenizer


class ModelLocal(ModelBase):
    record: ModelRecordLocalAny
    config: ModelConfigLocalAny

    @classmethod
    def new(cls, record: ModelRecordLocalAny, config: ModelConfigLocalAny) -> 'ModelLocal':
        return cls(
            tokenizer=try_get_tokenizer(record),
            record=record,
            config=config
        )

    @property
    def sampling_params(self) -> SamplingParams:
        return self.config.sampling_params or self.record.sampling_params

    @property
    def engine_params(self) -> EngineParamsAny:
        return self.config.engine_params or self.record.engine_params

    @property
    def urls(self) -> URLsLocalAny:
        assert isinstance(self.record.urls, URLsLocalAny)
        return self.record.urls


class ModelRemote(ModelBase):
    record: ModelRecordRemoteAny
    config: ModelConfigRemoteAny

    @classmethod
    def new(cls, record: ModelRecordRemoteAny, config: ModelConfigRemoteAny) -> 'ModelRemote':
        return cls(
            tokenizer=try_get_tokenizer(record),
            record=record,
            config=config
        )


    @property
    def sampling_params(self) -> SamplingParams:
        return self.config.sampling_params or self.record.sampling_params

    @property
    def urls(self) -> URLsRemoteAny:
        assert isinstance(self.record.urls, URLsRemoteAny)
        return self.record.urls


ModelAny = ModelLocal | ModelRemote


def try_resolve_record(c_model: ModelConfigAny, local_only: bool) -> Optional[ModelAny]:
    if isinstance(c_model, ModelConfigLocalAny):
        filtered_records = [r for r in RECORDS if isinstance(r, ModelRecordLocalAny)]
        record = next((r for r in filtered_records if r.resolve_name == c_model.model), None)
        if record is None:
            raise ValueError(f"Model {c_model.model} has no matching record in {filtered_records}; Was looking for resolve_name={c_model.model}")

        assert isinstance(record.urls, partial)
        record.urls = record.urls(url=c_model.base_url)

        return ModelLocal.new(record=record, config=c_model)

    elif isinstance(c_model, ModelConfigRemoteAny):
        if local_only:
            return None

        filtered_records = [r for r in RECORDS if isinstance(r, ModelRecordRemoteAny)]
        record = next((r for r in filtered_records if r.resolve_name == c_model.model), None)
        if record is None:
            raise ValueError(f"Model {c_model.model} has no matching record in {filtered_records}; Was looking for resolve_name={c_model.model}")

        assert isinstance(record.urls, partial)
        record.urls = record.urls(url=c_model.base_url)

        return ModelRemote.new(record=record, config=c_model)

    else:
        raise ValueError(f"Unknown model type: {type(c_model)}; Expected ModelConfigLocalAny or ModelConfigRemoteAny")
