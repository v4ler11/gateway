from functools import partial

from pydantic import BaseModel, ConfigDict, Field

from models.status import Status
from stt.models import ModelRecordAny, ModelConfigAny, URLsAny
from stt.models.records import RECORDS


class ModelBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Status = Field(default_factory=Status)


class Model(ModelBase):
    record: ModelRecordAny
    config: ModelConfigAny

    @classmethod
    def new(cls, record: ModelRecordAny, config: ModelConfigAny) -> "Model":
        return cls(
            record=record,
            config=config,
        )

    @property
    def urls(self) -> URLsAny:
        assert isinstance(self.record.urls, URLsAny)
        return self.record.urls


def try_resolve_record(c_model: ModelConfigAny) -> Model:
    record = next((r for r in RECORDS if r.resolve_name == c_model.model), None)
    if record is None:
        raise ValueError(
            f"Model {c_model.model} has no matching record in {RECORDS}; Was looking for resolve_name={c_model.model}")

    assert isinstance(record.urls, partial)
    record.urls = record.urls(url=c_model.base_url)

    return Model.new(record=record, config=c_model)
