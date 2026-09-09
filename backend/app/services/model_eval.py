"""模型性能测试任务：real_data 随机划分 -> 后台重训 A/B 变体 -> 测试集评估。

任务以线程方式在进程内执行，job_id 供前端轮询；结果存于内存字典
（原型阶段重启即失，符合当前无持久化需求）。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.config import get_settings  # noqa: F401  仅为触发 sys.path 注入


def variant_of(model_id: str) -> str:
    return "B" if "ngram" in model_id else "A"


def _eval_report_path() -> Path:
    return get_settings().model_dir / "eval_report.json"


def _read_report() -> Dict[str, object]:
    path = _eval_report_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_report(report: Dict[str, object]) -> None:
    path = _eval_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_metrics(model_id: str) -> Dict[str, object]:
    """返回最近一次模型测试写入的指标覆盖（无则空）。"""
    return dict(_read_report().get(model_id, {}))


class ModelEvalService:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, object]] = {}
        self._lock = threading.Lock()

    def submit(self, model_id: str, train_ratio: float, per_class_files: int,
               seed: int = 42) -> Dict[str, object]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "variant": variant_of(model_id),
            "train_ratio": float(train_ratio),
            "per_class_files": int(per_class_files),
            "status": "running",
            "created_at": time.time(),
            "error": None,
            "result": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _run(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        try:
            from ml.research.eval_pipeline import run_eval

            result = run_eval(
                variant=job["variant"],
                train_ratio=float(job["train_ratio"]),
                per_class_files=int(job["per_class_files"]),
                seed=42,
            )
            with self._lock:
                job["result"] = result
                job["status"] = "succeeded"
                job["elapsed_ms"] = int((time.time() - job["created_at"]) * 1000)
            self._persist(job)
        except Exception as exc:
            with self._lock:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["elapsed_ms"] = int((time.time() - job["created_at"]) * 1000)

    @staticmethod
    def _persist(job: Dict[str, object]) -> None:
        """把测试结果写入 eval_report.json，模型仓库页据此覆盖 accuracy / macro-F1 显示。"""
        result = job.get("result")
        if not result:
            return
        metrics = result["test_metrics"]
        report = _read_report()
        model_id = str(job["model_id"])
        report[model_id] = {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "train_ratio": float(job["train_ratio"]),
            "per_class_files": int(job["per_class_files"]),
            "variant": str(job["variant"]),
        }
        _write_report(report)


_service = ModelEvalService()


def get_eval_service() -> ModelEvalService:
    return _service
