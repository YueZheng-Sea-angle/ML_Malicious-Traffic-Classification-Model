"""推理服务：后端与算法层之间的唯一边界。

算法层不可用（未装 torch / numpy 报错等）时返回 mode="unavailable" 的
占位结果而非抛错，保证前端联调不被算法环境阻塞。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from app.config import get_settings  # noqa: F401  仅为触发 sys.path 注入


def _load_ml():
    from ml.predict import get_predictor

    return get_predictor


def inference_info() -> Dict[str, object]:
    try:
        return _load_ml()().info()
    except Exception as exc:
        return {
            "mode": "unavailable",
            "torch_available": False,
            "error": str(exc),
            "class_names": [],
        }


def run_inference(path: Path, max_flows: int = 64) -> Dict[str, object]:
    predictor = _load_ml()()
    return predictor.predict_file(Path(path), max_flows=max_flows)


def reload_predictor(checkpoint: Optional[Path] = None) -> Dict[str, object]:
    from ml.predict import get_predictor

    return get_predictor(checkpoint, reload=True).info()
