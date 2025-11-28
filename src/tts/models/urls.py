from models.urls import URLs


class URLsKokoro(URLs):
    @property
    def ping(self) -> str:
        raise NotImplementedError()

    @property
    def generate(self) -> str:
        raise NotImplementedError()
