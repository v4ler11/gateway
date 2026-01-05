from core import BASE_DIR


LOGS_DIR = BASE_DIR / "data" / "core" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE_DIR = BASE_DIR / ".models-cache"
TOKENIZERS_DIR = MODEL_CACHE_DIR / "tokenizers"
MODELS_DIR = MODEL_CACHE_DIR / "models"

YAML_CONFIG = BASE_DIR / "config.yaml"

PORT = 8000
