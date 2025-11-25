from core import BASE_DIR


LOGS_DIR = BASE_DIR / "data" / "inf_llamacpp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
