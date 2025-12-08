import asyncio

import uvicorn
import uvloop

from core import BASE_DIR
from core.logger import init_logger, info
from models.config import Config, models_from_config
from models.definitions import ModelTTSAny
from tts.inference.app import App
from tts.inference.grpc.server import grpc_server
from tts.inference.pipeline import init_pipeline


LOGS_DIR = BASE_DIR / "data" / "tts" / "logs"


async def start_services(models, pipeline, uvicorn_config):
    http_server = uvicorn.Server(uvicorn_config)

    info("Starting HTTP & gRPC servers...")

    await asyncio.gather(
        http_server.serve(),
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

    app = App.new(models, pipeline)

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=models[0].config.port,
        timeout_keep_alive=600,
        log_config=None
    )

    try:
        asyncio.run(start_services(models, pipeline, uvicorn_config))
    except KeyboardInterrupt:
        pass

    exit(0)
