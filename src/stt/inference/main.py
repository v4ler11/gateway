import asyncio

import uvicorn
import uvloop

from core import BASE_DIR
from core.logger import init_logger, info
from models.config import Config, models_from_config
from models.definitions import ModelSTTAny
from stt.inference.grpc.server import grpc_server
from stt.inference.rest.app import App
from stt.inference.streaming_parakeet import load_global_model

LOGS_DIR = BASE_DIR / "data" / "stt" / "logs"


async def start_services(g_server, uvicorn_config):
    http_server = uvicorn.Server(uvicorn_config)

    info("Starting HTTP & gRPC servers...")

    await asyncio.gather(
        http_server.serve(),
        g_server,
    )


def main():
    init_logger(LOGS_DIR)
    info("Logger initialized")

    config = Config.read_yaml()
    models = models_from_config(config)

    models = [m for m in models if isinstance(m, ModelSTTAny)]
    if len(models) > 1:
        raise RuntimeError("More than one TTS model is not supported")

    p_model = load_global_model()

    app = App.new(models, p_model)
    g_server = grpc_server(models, p_model)

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=models[0].config.port,
        timeout_keep_alive=600,
        log_config=None
    )

    try:
        asyncio.run(start_services(g_server, uvicorn_config))
    except KeyboardInterrupt:
        pass

    exit(0)
