"""模型管理：列表、激活、当前推理状态、性能测试。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import EvalJobResponse, EvalSubmitRequest, ModelInfo, ModelListResponse
from app.services.inference import inference_info
from app.services.model_eval import get_eval_service
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


@router.post("/evaluate", response_model=EvalJobResponse, status_code=202, summary="发起模型性能测试")
def start_evaluate(payload: EvalSubmitRequest) -> EvalJobResponse:
    model_id = payload.model_id or get_registry().active_id()
    if model_id in {None, "unavailable-baseline"}:
        raise HTTPException(status_code=400, detail="暂无可用权重，无法发起测试")
    job = get_eval_service().submit(model_id, payload.train_ratio, payload.per_class_files)
    return EvalJobResponse(**job)


@router.get("/evaluate/{job_id}", response_model=EvalJobResponse, summary="查询测试任务")
def get_evaluate(job_id: str) -> EvalJobResponse:
    job = get_eval_service().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"测试任务 {job_id} 不存在")
    return EvalJobResponse(**job)
