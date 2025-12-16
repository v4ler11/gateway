import asyncio
from typing import List, Any

from grpclib.server import Server

from core.logger import info
from stt.inference.grpc.proto_service_transcriptions import ProtoTranscriptionService
from models.definitions import ModelSTTAny


async def grpc_server(
        models: List[ModelSTTAny],
        p_model: Any,
):
    loop = asyncio.get_event_loop()

    async with Server(
            [
                ProtoTranscriptionService(
                    loop=loop,
                    p_model=p_model,
                    model=models[0],
                ),
            ]
    ) as server:

        await server.start("0.0.0.0", 50051)
        info("gRPC server started on 0.0.0.0:50051")

        await server.wait_closed()
