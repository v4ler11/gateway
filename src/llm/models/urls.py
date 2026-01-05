from models.urls import URLs


class URLsLlamaCpp(URLs):
    @property
    def ping(self) -> str:
        return f"{self.url}/health"

    @property
    def generate(self) -> str:
        return f"{self.url}/v1/chat/completions"


class URLsLmStudio(URLs):
    pass
