"""模型注册表：扫描 artifacts/models 下的权重并维护当前激活模型。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.services.inference import reload_predictor


class ModelRegistry:
    def __init__(self) -> None:
        self._active_id: Optional[str] = None

    def list_models(self) -> List[Dict[str, object]]:
        settings = get_settings()
        report = _read_feature_report(settings.model_dir / "feature_report.json")
        metrics = _read_metrics(settings.model_dir / "train_metrics.json")

        models: List[Dict[str, object]] = []
        for path in sorted(settings.model_dir.glob("*.pt")):
            model_id = path.stem
            models.append(
                {
                    "model_id": model_id,
                    "name": "MalFlowNet（CNN + BiLSTM + 门控统计特征）",
                    "version": "0.1.0",
                    "framework": "PyTorch",
                    "is_active": model_id == self.active_id(models_exist=True),
                    "accuracy": metrics.get("best_accuracy"),
                    "macro_f1": metrics.get("best_macro_f1"),
                    "trained_at": datetime.fromtimestamp(path.stat().st_mtime),
                    "selected_features": report.get("selected_names", []),
                    "description": f"权重文件 {path.name}",
                }
            )
        if not models:
            models.append(
                {
                    "model_id": "heuristic-baseline",
                    "name": "启发式基线（类别先验最近邻）",
                    "version": "0.0.1",
                    "framework": "NumPy",
                    "is_active": True,
                    "accuracy": None,
                    "macro_f1": None,
                    "trained_at": None,
                    "selected_features": [],
                    "description": "尚未训练权重时的演示模型，用于打通前后端链路",
                }
            )
        return models

    def active_id(self, models_exist: bool = False) -> Optional[str]:
        if self._active_id:
            return self._active_id
        settings = get_settings()
        weights = sorted(settings.model_dir.glob("*.pt"))
        if weights:
            return weights[0].stem
        return None if models_exist else "heuristic-baseline"

    def activate(self, model_id: str) -> Dict[str, object]:
        settings = get_settings()
        if model_id == "heuristic-baseline":
            self._active_id = model_id
            return reload_predictor(settings.model_dir / "__not_exist__.pt")
        path = settings.model_dir / f"{model_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"模型 {model_id} 不存在")
        self._active_id = model_id
        return reload_predictor(path)


def _read_feature_report(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    history = data.get("history", [])
    best_macro_f1 = max((h.get("macro_f1", 0.0) for h in history), default=None)
    return {"best_accuracy": data.get("best_accuracy"), "best_macro_f1": best_macro_f1}


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
