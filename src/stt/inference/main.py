from core import BASE_DIR
from core.logger import init_logger, info


LOGS_DIR = BASE_DIR / "data" / "stt" / "logs"


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    exit(0)
