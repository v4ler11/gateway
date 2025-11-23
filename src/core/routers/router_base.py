from fastapi import APIRouter

from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference


class HealthResponse(BaseModel):
    status: str = "healthy"


class BaseRouter(APIRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_api_route(
            "/docs",
            self._scalar,
            methods=["GET"],
            include_in_schema=False
        )
        self.add_api_route(
            "/health",
            self.health,
            methods=["GET"],
            include_in_schema=True,
            responses={
                200: dict(
                    description="Health check",
                    model=HealthResponse,
                )
            }
        )

    async def _scalar(self):
        return get_scalar_api_reference(
            openapi_url="/v1/openapi.json",
            title="",
        )

    async def health(self):
        return HealthResponse()
