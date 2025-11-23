from core import BASE_DIR


LOGS_DIR = BASE_DIR / "data" / "core" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


YAML_CONFIG = BASE_DIR / "config.yaml"
if not YAML_CONFIG.is_file():
    raise Exception(
        f"config.yaml doesn't exist, create one using config.example.yaml as a reference; "
        f"seeing files: {', '.join([str(d) for d in BASE_DIR.iterdir()])}"
    )


PORT = 8000
