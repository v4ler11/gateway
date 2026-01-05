import asyncio

import uvloop
import uvicorn

from core import BASE_DIR
from core.app import App
from core.globals import LOGS_DIR, PORT, YAML_CONFIG
from core.logger import init_logger, info
from core.status.worker import spawn_worker as spawn_status_worker
from models.config import Config, models_from_config


def checks():
    if not YAML_CONFIG.is_file():
        raise Exception(
            f"config.yaml doesn't exist, create one using config.example.yaml as a reference; "
            f"seeing files: {', '.join([str(d) for d in BASE_DIR.iterdir()])}"
        )


def main():
    checks()

    init_logger(LOGS_DIR)
    info("Logger initialized")

    config = Config.read_yaml()
    models = models_from_config(config)

    w_status = spawn_status_worker(models)

    app = App.new(models)

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

    w_status.stop_event.set()

    exit(0)
