from llm.models import (
    ModelConfigAny as ModelConfigLLMAny,
    MODEL_CONFIG_CLASSES as MODEL_CONFIG_CLASSES_LLM,
)
from llm.models.models import ModelAny as ModelLLMAny
from tts.models import (
    ModelConfigAny as ModelConfigTTSAny,
    MODEL_CONFIG_CLASSES as MODEL_CONFIG_CLASSES_TTS,
)
from tts.models.models import Model as ModelTTSAny


ModelConfigLLMAny = ModelConfigLLMAny
ModelConfigTTSAny = ModelConfigTTSAny

ModelLLMAny = ModelLLMAny
ModelTTSAny = ModelTTSAny
ModelAny = ModelLLMAny | ModelTTSAny

ModelConfigAny = ModelConfigLLMAny | ModelConfigTTSAny
MODEL_CONFIG_CLASSES = [*MODEL_CONFIG_CLASSES_LLM, *MODEL_CONFIG_CLASSES_TTS]
