import asyncio

import uvloop
import uvicorn

from core.app import App
from core.globals import LOGS_DIR, PORT
from core.logger import init_logger, info
from models.read_config import read_config


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    config = read_config()
    print(config)

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
