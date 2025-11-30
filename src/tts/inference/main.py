import asyncio

import uvicorn
import uvloop

from core import BASE_DIR
from core.logger import init_logger, info
from models.config import Config, models_from_config
from models.definitions import ModelTTSAny
from tts.inference.app import App


LOGS_DIR = BASE_DIR / "data" / "tts" / "logs"


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    config = Config.read_yaml()
    models = models_from_config(config)

    models = [m for m in models if isinstance(m, ModelTTSAny)]
    if len(models) > 1:
        raise RuntimeError("More than one TTS model is not supported")

    app = App.new(models)

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=models[0].config.port,
        timeout_keep_alive=600,
        log_config=None
    )
    server = uvicorn.Server(config)
    server.run()

    exit(0)
