import time
from typing import Literal, List

from pydantic import BaseModel
from starlette import status

from core.routers.router_base import BaseRouter
from models.s3_models.models import ModelAny
from models.s3_models.status import Status


class ModelResponse(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    status: Status


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"


class ModelsRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelAny],
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.models = models

        self.add_api_route(
            "/v0/models",
            self._models,
            methods=["GET"],
            status_code=status.HTTP_200_OK,
            responses={
                200: dict(
                    description="Returns a list of models",
                    model=ModelsResponse
                )
            }
        )

    async def _models(self):
        return [
            ModelResponse(
                id=m.record.resolve_name,
                created=int(time.time()),
                status=m.status,
            )
            for m in self.models
        ]
