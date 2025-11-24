from functools import partial
from types import FunctionType

from pydantic import BaseModel

from models.s1_records import ModelRecordLocalAny, ModelRecordRemoteAny
from models.s1_records.records import RECORDS
from models.s2_from_config import ModelConfigLocalAny, ModelConfigRemoteAny, ModelConfigAny
from models.s2_from_config.config import Config


class ModelLocal(BaseModel):
    record: ModelRecordLocalAny
    config: ModelConfigLocalAny


class ModelRemote(BaseModel):
    record: ModelRecordRemoteAny
    config: ModelConfigRemoteAny


ModelAny = ModelLocal | ModelRemote


# todo: simplify me
def try_resolve_record(c_model: ModelConfigAny) -> ModelAny:
    if isinstance(c_model, ModelConfigLocalAny):
        filtered_records = [r for r in RECORDS if isinstance(r, ModelRecordLocalAny)]
        record = next((r for r in filtered_records if r.resolve_name == c_model.model), None)
        if record is None:
            raise ValueError(f"Model {c_model.model} has no matching record in {filtered_records}; Was looking for resolve_name={c_model.model}")

        assert isinstance(record.urls, partial)
        record.urls = record.urls(url=c_model.base_url)

        return ModelLocal(record=record, config=c_model)

    elif isinstance(c_model, ModelConfigRemoteAny):
        filtered_records = [r for r in RECORDS if isinstance(r, ModelRecordRemoteAny)]
        record = next((r for r in filtered_records if r.resolve_name == c_model.model), None)
        if record is None:
            raise ValueError(f"Model {c_model.model} has no matching record in {filtered_records}; Was looking for resolve_name={c_model.model}")

        assert isinstance(record.urls, partial)
        record.urls = record.urls(url=c_model.base_url)

        return ModelRemote(record=record, config=c_model)

    else:
        raise ValueError(f"Unknown model type: {type(c_model)}; Expected ModelConfigLocalAny or ModelConfigRemoteAny")


def models_from_config(config: Config) -> list[ModelAny]:
    return [try_resolve_record(c_model) for c_model in config.models]

