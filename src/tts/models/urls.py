from models.urls import URLs


class URLsKokoro(URLs):
    @property
    def ping(self) -> str:
        return f"{self.url}/health"

    @property
    def generate(self) -> str:
        return f"{self.url}/v1/audio/stream"
