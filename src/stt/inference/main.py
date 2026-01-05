import asyncio

import onnx_asr
import uvloop

from core import BASE_DIR
from core.logger import init_logger, info
from models.config import Config, models_from_config
from models.definitions import ModelSTTAny
from stt.inference.grpc.server import grpc_server
import onnxruntime as rt


LOGS_DIR = BASE_DIR / "data" / "stt" / "logs"


def patched_get_providers():
    providers = rt.get_available_providers()
    if 'TensorrtExecutionProvider' in providers:
        providers.remove('TensorrtExecutionProvider')
    return providers


def load_models():
    rt.get_available_providers = patched_get_providers
    sess_opts = rt.SessionOptions()

    info(f"Loading nemo-parakeet-tdt-0.6b-v3...")
    parakeet_model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        sess_options=sess_opts, providers=["CUDAExecutionProvider"]
    )

    info(f"Loading silero...")
    vad_model = onnx_asr.load_vad(
        "silero",
        sess_options=sess_opts, providers=["CUDAExecutionProvider"]
    )
    info(f"Loading models OK")

    return parakeet_model, vad_model


async def start_services(g_server):
    info("Starting gRPC server")

    await asyncio.gather(
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

    parakeet_model, vad_model = load_models()

    g_server = grpc_server(
        models=models,
        parakeet_model=parakeet_model,
        vad_model=vad_model,
    )

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    try:
        asyncio.run(start_services(g_server))
    except KeyboardInterrupt:
        pass

    exit(0)
