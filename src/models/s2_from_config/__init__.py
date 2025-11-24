from models.s2_from_config.model_config import ModelConfigRemoteLmStudio, ModelConfigLocalLlamaCpp


ModelConfigLocalAny = ModelConfigLocalLlamaCpp
ModelConfigRemoteAny = ModelConfigRemoteLmStudio
ModelConfigAny = ModelConfigLocalAny | ModelConfigRemoteAny

MODEL_CLASSES = [ModelConfigRemoteLmStudio, ModelConfigLocalLlamaCpp]
