from pydantic import BaseModel, ConfigDict, Field

from models.status import Status
from stt.models import ModelRecordAny, ModelConfigAny
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


def try_resolve_record(c_model: ModelConfigAny) -> Model:
    record = next((r for r in RECORDS if r.resolve_name == c_model.model), None)
    if record is None:
        raise ValueError(
            f"Model {c_model.model} has no matching record in {RECORDS}; Was looking for resolve_name={c_model.model}")

    return Model.new(record=record, config=c_model)
