from grpclib.server import Server
from grpclib.utils import graceful_exit

from src.generated.greeter import GreeterBase, HelloRequest, HelloReply


class GreeterService(GreeterBase):
    async def say_hello(self, request: HelloRequest) -> HelloReply:
        print(f"Received request from: {request.name}")
        return HelloReply(message=f"Hello, {request.name}!")


async def main_async():
    server = Server([GreeterService()])

    with graceful_exit([server]):
        await server.start("127.0.0.1", 50051)
        print("Server started on 127.0.0.1:50051")
        await server.wait_closed()
