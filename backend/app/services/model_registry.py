"""模型注册表：扫描 artifacts/models 下的权重并维护当前激活模型。

每个 .pt 的展示信息（类别集、验证指标、是否 n-gram、特征入选）直接取自
checkpoint 内嵌元数据，避免共享全局 json 造成的「多模型显示同一指标」误解。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.services.inference import reload_predictor
from app.services.model_eval import latest_metrics

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    TORCH_AVAILABLE = False

BASE_MODEL_ID = "malflow_datacon_tools"  # T1 A 组基线（默认激活）
NGRAM_MODEL_ID = f"{BASE_MODEL_ID}_ngram"  # T1 B 组 n-gram 增强


class ModelRegistry:
    def __init__(self) -> None:
        self._active_id: Optional[str] = None

    def list_models(self) -> List[Dict[str, object]]:
        models: List[Dict[str, object]] = []
        for path in self._sorted_weights():
            meta = _read_checkpoint_meta(path)
            model_id = path.stem
            eval_ov = latest_metrics(model_id)
            is_ngram = bool((meta or {}).get("ngram_dim")) if meta else "ngram" in model_id
            kind = (meta or {}).get("kind", "torch") if meta else "torch"
            framework = "LightGBM" if kind == "lightgbm" else "PyTorch"
            group = "B · n-gram 增强" if is_ngram else "A · 基线"
            description = _describe(path.stem, is_ngram, meta)
            if eval_ov:
                description += " · 指标来自最近一次模型测试"
            models.append(
                {
                    "model_id": model_id,
                    "name": f"T1 · {group}（{framework}）",
                    "version": str((meta or {}).get("version", "0.1.0")),
                    "framework": framework,
                    "is_active": model_id == self.active_id(models_exist=True),
                    "accuracy": eval_ov.get("accuracy", meta.get("accuracy") if meta else None),
                    "macro_f1": eval_ov.get("macro_f1", meta.get("macro_f1") if meta else None),
                    "trained_at": datetime.fromtimestamp(path.stat().st_mtime),
                    "selected_features": meta.get("selected_features", []) if meta else [],
                    "description": description,
                }
            )
        if not models:
            models.append(
                {
                    "model_id": "unavailable-baseline",
                    "name": "暂无可用权重（推理不可用）",
                    "version": "0.0.1",
                    "framework": "—",
                    "is_active": True,
                    "accuracy": None,
                    "macro_f1": None,
                    "trained_at": None,
                    "selected_features": [],
                    "description": "artifacts/models 下没有 .pt 权重，推理返回占位结果",
                }
            )
        return models

    def _sorted_weights(self) -> List[Path]:
        # T1 A 组基线默认激活，n-gram（B 组）排后
        weights = self.model_dir_weights()
        return sorted(weights, key=lambda p: (p.stem != BASE_MODEL_ID, "ngram" in p.stem, p.name))

    def model_dir_weights(self) -> List[Path]:
        return list(get_settings().model_dir.glob("*.pt"))

    def active_id(self, models_exist: bool = False) -> Optional[str]:
        if self._active_id:
            return self._active_id
        weights = self._sorted_weights()
        if weights:
            return weights[0].stem
        return None if models_exist else "unavailable-baseline"

    def activate(self, model_id: str) -> Dict[str, object]:
        settings = get_settings()
        if model_id == "unavailable-baseline":
            self._active_id = model_id
            return reload_predictor(settings.model_dir / "__not_exist__.pt")
        path = settings.model_dir / f"{model_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"模型 {model_id} 不存在")
        self._active_id = model_id
        return reload_predictor(path)


def _describe(model_id: str, is_ngram: bool, meta: Optional[Dict[str, object]]) -> str:
    n_classes = len((meta or {}).get("class_names", [])) if meta else 0
    kind = (meta or {}).get("kind", "torch") if meta else "torch"
    if kind == "lightgbm":
        route = "统计特征 + 载荷字节 n-gram" if is_ngram else "标准化 40 维统计特征"
        engine = "LightGBM（多分类）"
    else:
        route = "三路 + 载荷字节 n-gram" if is_ngram else "三路输入（统计 + 包长 + 字节）"
        engine = "MalFlowNet（PyTorch）"
    return (
        f"{model_id}.pt · {engine} · 特征：{route} · {n_classes} 类代理/隧道工具"
    )


def _read_checkpoint_meta(path: Path) -> Optional[Dict[str, object]]:
    if not TORCH_AVAILABLE:
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    metrics = payload.get("metrics") or {}
    return {
        "kind": payload.get("kind", "torch"),
        "version": payload.get("version"),
        "class_names": list(payload.get("class_names", [])),
        "ngram_dim": int(payload.get("ngram_dim", 0) or 0),
        "selected_features": list(payload.get("selected_features", [])),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
    }


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
