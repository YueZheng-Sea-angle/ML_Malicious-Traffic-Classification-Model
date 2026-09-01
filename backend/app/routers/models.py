"""模型管理：列表、激活、当前推理状态。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import ModelInfo, ModelListResponse
from app.services.inference import inference_info
from app.services.model_registry import get_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse, summary="模型列表")
def list_models() -> ModelListResponse:
    registry = get_registry()
    items = registry.list_models()
    return ModelListResponse(
        total=len(items),
        active_model_id=registry.active_id(),
        items=[ModelInfo(**item) for item in items],
    )


@router.get("/runtime", summary="当前推理运行状态")
def runtime() -> dict:
    return inference_info()


@router.post("/{model_id}/activate", summary="激活指定模型")
def activate(model_id: str) -> dict:
    try:
        info = get_registry().activate(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"activated": model_id, "runtime": info}
