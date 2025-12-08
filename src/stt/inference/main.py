import asyncio

from core import BASE_DIR
from core.logger import init_logger, info
from stt.inference.grpc_app import main_async


LOGS_DIR = BASE_DIR / "data" / "stt" / "logs"


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    asyncio.run(main_async())

    exit(0)
