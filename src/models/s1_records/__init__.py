from models.s1_records.engine_params import EngineParamsLlamacpp
from models.s1_records.model_record import ModelRecordLlamaCpp, ModelRecordLMStudio, SamplingParams
from models.s1_records.urls import URLsLlamaCpp, URLsLmStudio


ModelRecordLocalAny = ModelRecordLlamaCpp
ModelRecordRemoteAny = ModelRecordLMStudio
ModelRecordAny = ModelRecordLocalAny | ModelRecordRemoteAny

EngineParamsAny = EngineParamsLlamacpp

URLsLocalAny = URLsLlamaCpp
URLsRemoteAny = URLsLmStudio
URLsAny = URLsLocalAny | URLsRemoteAny
