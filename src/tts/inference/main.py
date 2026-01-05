import asyncio

from core import BASE_DIR
from core.logger import init_logger, info
from models.config import Config, models_from_config
from models.definitions import ModelTTSAny
from tts.inference.grpc.server import grpc_server
from tts.inference.pipeline import init_pipeline


LOGS_DIR = BASE_DIR / "data" / "tts" / "logs"


async def start_services(models, pipeline):
    await asyncio.gather(
        grpc_server(models, pipeline),
    )


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    config = Config.read_yaml()
    models = models_from_config(config)

    models = [m for m in models if isinstance(m, ModelTTSAny)]
    if len(models) > 1:
        raise RuntimeError("More than one TTS model is not supported")

    pipeline = init_pipeline(models[0])

    try:
        asyncio.run(start_services(models, pipeline))
    except KeyboardInterrupt:
        pass

    exit(0)
