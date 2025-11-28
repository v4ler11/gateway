import time
from typing import Literal, List

from pydantic import BaseModel
from starlette import status

from core.routers.router_base import BaseRouter
from core.routers.schemas import ErrorResponse, error_constructor
from models.definitions import ModelAny
from models.status import Status


class ModelResponse(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    status: Status


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelResponse]


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
                ),
                500: dict(
                    description="Internal server error",
                    model=ErrorResponse,
                ),
            }
        )

    async def _models(self):
        try:
            return ModelsResponse(
                data=[
                    ModelResponse(
                        id=m.record.resolve_name,
                        created=int(time.time()),
                        status=m.status,
                    )
                    for m in self.models
                ]
            )
        except Exception as e:
            return error_constructor(
                message=f"Internal server error: {str(e)}",
                error_type="internal_server_error",
                status_code=500
            )
