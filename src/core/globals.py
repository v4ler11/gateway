from core import BASE_DIR


LOGS_DIR = BASE_DIR / "data" / "core" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8000
