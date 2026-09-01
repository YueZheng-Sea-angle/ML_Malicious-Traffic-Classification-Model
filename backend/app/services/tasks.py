"""任务调度：接收任务 -> 后台执行推理 -> 回写结果。

原型阶段使用 FastAPI BackgroundTasks（同进程线程池），接口形态与后续
换成 Celery/RQ 时一致：提交后立即返回 task_id，前端轮询状态。
"""

from __future__ import annotations

import time
from datetime import datetime

from app.services.inference import run_inference
from app.services.storage import TaskRecord, get_store


def execute_task(task_id: str) -> None:
    store = get_store()
    task = store.get_task(task_id)
    if task is None:
        return
    file_record = store.get_file(task.file_id)
    if file_record is None:
        store.update_task(task_id, status="failed", error="上传文件记录不存在",
                          finished_at=datetime.now())
        return

    store.update_task(task_id, status="running")
    started = time.perf_counter()
    try:
        result = run_inference(file_record.path, max_flows=task.max_flows)
        store.update_task(
            task_id,
            status="succeeded",
            result=result,
            finished_at=datetime.now(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        store.update_task(
            task_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )


def task_to_dict(task: TaskRecord) -> dict:
    return {
        "task_id": task.task_id,
        "file_id": task.file_id,
        "filename": task.filename,
        "status": task.status,
        "model_id": task.model_id,
        "created_at": task.created_at,
        "finished_at": task.finished_at,
        "elapsed_ms": task.elapsed_ms,
        "error": task.error,
        "result": _result_to_dict(task.result),
    }


def _result_to_dict(result: dict | None) -> dict | None:
    if not result:
        return None
    return {
        "label": result.get("label", "unknown"),
        "label_zh": result.get("label_zh", "未知"),
        "confidence": result.get("confidence", 0.0),
        "malicious_score": result.get("malicious_score", 0.0),
        "probabilities": result.get("probabilities", {}),
        "flow_count": result.get("flow_count", 0),
        "flows": result.get("flows", []),
        "top_features": result.get("top_features", []),
        "mode": result.get("mode", "unknown"),
    }
