import time
from typing import Literal, List

from pydantic import BaseModel
from starlette import status

from core.routers.router_base import BaseRouter
from core.routers.schemas import ErrorResponse, error_constructor
from models.s3_models.models import ModelAny


class OAIModelsResponseModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "system"

    @classmethod
    def new(cls, model: str) -> "OAIModelsResponseModel":
        return cls(
            id=model,
            created=int(time.time())
        )


class OAIModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[OAIModelsResponseModel]


class OAIModelsRouter(BaseRouter):
    def __init__(
            self,
            models: List[ModelAny],
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.models = models

        self.add_api_route(
            "/oai/v1/models",
            self._models,
            methods=["GET"],
            status_code=status.HTTP_200_OK,
            responses={
                200: dict(
                    description="Returns running LLMs in OpenAI format",
                    model=OAIModelsResponse,
                ),
                500: dict(
                    description="Internal server error",
                    model=ErrorResponse,
                ),
            }
        )

    async def _models(self):
        try:
            return OAIModelsResponse(
                data=[
                    OAIModelsResponseModel.new(m.record.resolve_name)
                    for m in self.models
                    if m.status.running
                ]
            )
        except Exception as e:
            return error_constructor(
                message=f"Internal server error: {str(e)}",
                error_type="internal_server_error",
                status_code=500
            )
