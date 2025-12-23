from typing import List

from grpclib.server import Server
from kokoro import KPipeline

from core.logger import info
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

        await server.start("0.0.0.0", 50101)
        info("gRPC server started on 0.0.0.0:50101")

        await server.wait_closed()
