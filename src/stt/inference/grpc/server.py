import asyncio
from typing import List, Any

from grpclib.server import Server

from core.logger import info
from stt.globals import GRPC_PORT
from stt.inference.grpc.proto_service_transcriptions import ProtoTranscriptionService
from models.definitions import ModelSTTAny


async def grpc_server(
        models: List[ModelSTTAny],
        parakeet_model: Any,
        vad_model: Any,
):
    loop = asyncio.get_event_loop()

    async with Server(
            [
                ProtoTranscriptionService(
                    loop=loop,
                    model=models[0],
                    parakeet_model=parakeet_model,
                    vad_model=vad_model,
                ),
            ]
    ) as server:

        await server.start("0.0.0.0", GRPC_PORT)
        info(f"gRPC server started on 0.0.0.0:{GRPC_PORT}")

        await server.wait_closed()
