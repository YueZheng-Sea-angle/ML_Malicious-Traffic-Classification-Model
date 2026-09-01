"""分类任务：创建、查询、列表、统计。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.schemas import (
    StatsResponse,
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
)
from app.services.storage import get_store
from app.services.tasks import execute_task, task_to_dict

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=202, summary="提交分类任务")
def create_task(payload: TaskCreateRequest, background: BackgroundTasks) -> TaskResponse:
    store = get_store()
    file_record = store.get_file(payload.file_id)
    if file_record is None:
        raise HTTPException(status_code=404, detail=f"文件 {payload.file_id} 不存在，请先上传")

    task = store.add_task(file_record, payload.model_id, payload.max_flows, payload.note)
    background.add_task(execute_task, task.task_id)
    return TaskResponse(**task_to_dict(task))


@router.get("", response_model=TaskListResponse, summary="任务列表")
def list_tasks(
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None, description="pending/running/succeeded/failed"),
) -> TaskListResponse:
    tasks = get_store().list_tasks(limit=limit, status=status)
    return TaskListResponse(
        total=len(tasks), items=[TaskResponse(**task_to_dict(t)) for t in tasks]
    )


@router.get("/stats", response_model=StatsResponse, summary="任务统计（用于仪表盘）")
def stats() -> StatsResponse:
    tasks = get_store().list_tasks(limit=500)
    distribution: dict = {}
    elapsed = []
    for task in tasks:
        if task.result:
            label = str(task.result.get("label_zh") or task.result.get("label"))
            distribution[label] = distribution.get(label, 0) + 1
        if task.elapsed_ms:
            elapsed.append(task.elapsed_ms)
    return StatsResponse(
        total_tasks=len(tasks),
        succeeded=sum(1 for t in tasks if t.status == "succeeded"),
        failed=sum(1 for t in tasks if t.status == "failed"),
        running=sum(1 for t in tasks if t.status in {"pending", "running"}),
        label_distribution=distribution,
        average_elapsed_ms=round(sum(elapsed) / len(elapsed), 2) if elapsed else 0.0,
    )


@router.get("/{task_id}", response_model=TaskResponse, summary="任务详情与分类结果")
def get_task(task_id: str) -> TaskResponse:
    task = get_store().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return TaskResponse(**task_to_dict(task))
