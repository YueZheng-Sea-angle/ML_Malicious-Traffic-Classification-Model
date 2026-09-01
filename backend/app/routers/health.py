"""健康检查：前端启动页与环境验收依赖此接口。"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse
from app.services.inference import inference_info

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="服务健康检查")
def health() -> HealthResponse:
    settings = get_settings()
    info = inference_info()
    return HealthResponse(
        app_name=settings.app_name,
        version=settings.version,
        inference_mode=str(info.get("mode", "unknown")),
        torch_available=bool(info.get("torch_available", False)),
    )
