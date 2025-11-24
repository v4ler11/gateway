import asyncio

import uvloop
import uvicorn

from core.app import App
from core.globals import LOGS_DIR, PORT
from core.logger import init_logger, info
from models.s2_from_config.config import Config
from models.s3_models.models import models_from_config


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    config = Config.read()
    models = models_from_config(config)
    print(models)

    app = App.new()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=PORT,
        timeout_keep_alive=600,
        log_config=None
    )
    server = uvicorn.Server(config)
    server.run()

    exit(0)
