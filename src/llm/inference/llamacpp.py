import os
import subprocess
import time
from pathlib import Path

from core import BASE_DIR
from core.globals import MODELS_DIR
from core.logger import init_logger, info
from hf.download import download_repo_paths
from llm.models import ModelRecordLlamaCpp
from llm.models.models import ModelLocal
from models.config import Config, models_from_config


LOGS_DIR = BASE_DIR / "data" / "inf_llamacpp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_model() -> ModelLocal:
    config = Config.read_yaml()
    models = models_from_config(config)
    models = [m for m in models if isinstance(m.record, ModelRecordLlamaCpp)]

    if not models:
        raise ValueError("No models found to run")
    if len(models) > 1:
        raise ValueError("Multiple models are not supported yet")

    model = models[0]
    assert isinstance(model, ModelLocal)
    return model


def download_model(model: ModelLocal) -> Path:
    d_files, err = download_repo_paths(
        model.record.model,
        MODELS_DIR / model.record.model.replace("/", "_"),
        [model.record.model_file]
    )
    if err is not None:
        raise ValueError(f"Failed to download model {model.record.model}: {err}")

    model_file = d_files.get(model.record.model_file)
    if not model_file:
        raise ValueError(f"Failed to download model {model.record.model}: {err}")

    return model_file


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    model = get_model()
    print(1)

    model_file = download_model(model)
    print(2)

    cmd = [
        "/app/llama-server",
        "-m", str(model_file),
        *model.engine_params.model_dump_to_args(),
        "--host", "0.0.0.0",
        "--port", str(model.config.port),
    ]
    info(f"Starting Llama.cpp with following command:\n{' '.join(cmd)}")

    env = os.environ.copy()
    process = subprocess.Popen(cmd, env=env)

    try:
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print("Process has terminated")
                break

    except KeyboardInterrupt:
        print("Terminating process...")
        process.terminate()
        process.wait()
