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

from stt.models import (
    ModelConfigAny as ModelConfigSTTAny,
    MODEL_CONFIG_CLASSES as MODEL_CONFIG_CLASSES_STT,
)
from stt.models.models import Model as ModelSTTAny


# ModelConfigLLMAny = ModelConfigLLMAny
# ModelConfigTTSAny = ModelConfigTTSAny

# ModelLLMAny = ModelLLMAny
# ModelTTSAny = ModelTTSAny
ModelAny = ModelLLMAny | ModelTTSAny | ModelSTTAny

ModelConfigAny = (ModelConfigLLMAny | ModelConfigTTSAny | ModelConfigSTTAny)
MODEL_CONFIG_CLASSES = [
    *MODEL_CONFIG_CLASSES_LLM,
    *MODEL_CONFIG_CLASSES_TTS,
    *MODEL_CONFIG_CLASSES_STT,
]
