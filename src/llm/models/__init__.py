from llm.models.engine_params import EngineParamsLlamacpp
from llm.models.model_config import ModelConfigLocalLlamaCpp, ModelConfigRemoteLmStudio
from llm.models.model_record import ModelRecordLlamaCpp, ModelRecordLMStudio, SamplingParams
from llm.models.urls import URLsLlamaCpp, URLsLmStudio

ModelRecordLocalAny = ModelRecordLlamaCpp
ModelRecordRemoteAny = ModelRecordLMStudio
ModelRecordAny = ModelRecordLocalAny | ModelRecordRemoteAny

EngineParamsAny = EngineParamsLlamacpp

URLsLocalAny = URLsLlamaCpp
URLsRemoteAny = URLsLmStudio
URLsAny = URLsLocalAny | URLsRemoteAny

ModelConfigLocalAny = ModelConfigLocalLlamaCpp
ModelConfigRemoteAny = ModelConfigRemoteLmStudio
ModelConfigAny = ModelConfigLocalAny | ModelConfigRemoteAny

MODEL_CLASSES = [ModelConfigRemoteLmStudio, ModelConfigLocalLlamaCpp]
