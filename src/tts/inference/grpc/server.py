from typing import List

from grpclib.server import Server
from kokoro import KPipeline

from core.logger import info
from tts.globals import GRPC_PORT
from tts.inference.grpc.proto_service_audio import ProtoAudioService
from models.definitions import ModelTTSAny


async def grpc_server(
        models: List[ModelTTSAny],
        pipeline: KPipeline
):
    async with Server(
            [
                ProtoAudioService(
                    model=models[0],
                    pipeline=pipeline,
                ),
            ]
    ) as server:

        await server.start("0.0.0.0", GRPC_PORT)
        info(f"gRPC server started on 0.0.0.0:{GRPC_PORT}")

        await server.wait_closed()
