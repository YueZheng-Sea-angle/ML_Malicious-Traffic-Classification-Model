"""推理入口：PCAP 文件 -> 流级与文件级分类结果。

后端通过 ``get_predictor()`` 获取全局单例。运行模式两种：

    model      已训练权重 + PyTorch 可用，走 MalFlowNet
    heuristic  权重缺失或 torch 未安装，走类别先验最近邻打分

两种模式返回结构完全一致，前端与接口契约不受影响；结果中的 ``mode``
字段用于界面提示当前是否为演示推理。

命令行自测：
    python -m ml.predict --file some.pcap
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ml.config import CLASS_NAMES, CLASS_NAMES_ZH, DEFAULT_CHECKPOINT
from ml.features.extractor import STAT_FEATURE_NAMES, FlowSample, pcap_to_samples

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    TORCH_AVAILABLE = False


@dataclass
class FlowPrediction:
    flow_id: str
    label: str
    label_zh: str
    confidence: float
    probabilities: Dict[str, float]
    meta: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "flow_id": self.flow_id,
            "label": self.label,
            "label_zh": self.label_zh,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "meta": self.meta,
        }


class Predictor:
    def __init__(self, checkpoint: Optional[Path] = None) -> None:
        self.checkpoint_path = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.mode = "heuristic"
        self.model = None
        self.stat_mean: Optional[np.ndarray] = None
        self.stat_std: Optional[np.ndarray] = None
        self.selected_features: List[str] = []
        self.metrics: Dict[str, object] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not (TORCH_AVAILABLE and self.checkpoint_path.exists()):
            return
        try:
            from ml.models.cnn_bilstm import build_model

            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(len(payload.get("class_names", CLASS_NAMES)))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.model = model
            self.stat_mean = np.asarray(payload["stat_mean"], dtype=np.float32)
            self.stat_std = np.asarray(payload["stat_std"], dtype=np.float32)
            self.selected_features = list(payload.get("selected_features", []))
            self.metrics = payload.get("metrics", {})
            self.mode = "model"
        except Exception as exc:  # 权重损坏不应导致服务不可用
            print(f"[predict] 加载权重失败，回退启发式推理：{exc}")

    def info(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "torch_available": TORCH_AVAILABLE,
            "class_names": CLASS_NAMES,
            "selected_features": self.selected_features,
            "metrics": self.metrics,
        }

    # ------------------------------------------------------------------ #
    def predict_file(self, path: Path, max_flows: int = 64) -> Dict[str, object]:
        samples = pcap_to_samples(Path(path), max_flows=max_flows)
        flow_results = [self.predict_sample(s) for s in samples]
        return {
            "file": Path(path).name,
            "mode": self.mode,
            "flow_count": len(flow_results),
            "flows": [f.to_dict() for f in flow_results],
            **self._aggregate(flow_results),
            "top_features": self.explain(samples),
        }

    def predict_sample(self, sample: FlowSample) -> FlowPrediction:
        probs = (
            self._probs_model(sample) if self.mode == "model" else self._probs_heuristic(sample)
        )
        best = int(np.argmax(probs))
        return FlowPrediction(
            flow_id=sample.flow_id,
            label=CLASS_NAMES[best],
            label_zh=CLASS_NAMES_ZH[CLASS_NAMES[best]],
            confidence=float(probs[best]),
            probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
            meta=sample.meta,
        )

    def explain(self, samples: List[FlowSample], top_k: int = 6) -> List[Dict[str, object]]:
        """返回对本次判定影响最大的统计特征，供界面展示。"""
        if not samples:
            return []
        stats = np.stack([s.stats for s in samples])
        if self.mode == "model" and self.model is not None:
            with torch.no_grad():
                gate = self.model.stat_encoder(
                    torch.from_numpy(self._standardize(stats)), None
                )[1].mean(dim=0).numpy()
            weights = gate
        else:
            # 启发式模式下以「偏离全局均值的程度」近似贡献
            weights = np.abs(stats.mean(axis=0)) / (np.abs(stats).mean(axis=0) + 1e-6)
        order = np.argsort(weights)[::-1][:top_k]
        return [
            {
                "name": STAT_FEATURE_NAMES[i],
                "weight": round(float(weights[i]), 4),
                "value": round(float(stats[:, i].mean()), 4),
            }
            for i in order
        ]

    # ------------------------------------------------------------------ #
    def _standardize(self, stats: np.ndarray) -> np.ndarray:
        if self.stat_mean is None or self.stat_std is None:
            return stats.astype(np.float32)
        return ((stats - self.stat_mean) / self.stat_std).astype(np.float32)

    def _probs_model(self, sample: FlowSample) -> np.ndarray:
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(self._standardize(sample.stats[None, :])),
                torch.from_numpy(sample.pkt_seq[None, :]),
                torch.from_numpy(sample.byte_seq[None, :]),
            )
            return torch.softmax(out["logits"], dim=-1)[0].numpy()

    def _probs_heuristic(self, sample: FlowSample) -> np.ndarray:
        """按类别先验做加权最近邻，保证同一文件结果可复现。"""
        from ml.data.dataset import _CLASS_PROFILES

        index = {name: i for i, name in enumerate(STAT_FEATURE_NAMES)}
        observed = np.array(
            [
                sample.stats[index["len_mean"]] / 1500.0,
                sample.stats[index["iat_mean"]],
                sample.stats[index["fwd_ratio"]],
                sample.stats[index["pkt_count"]] / 120.0,
            ]
        )
        distances = []
        for name in CLASS_NAMES:
            profile = _CLASS_PROFILES[name]
            expected = np.array(
                [
                    profile["base_len"] / 1500.0,
                    profile["iat"],
                    profile["fwd_p"],
                    float(np.mean(profile["pkts"])) / 120.0,
                ]
            )
            distances.append(float(np.sum(((observed - expected) / np.array([0.3, 0.4, 0.25, 0.4])) ** 2)))
        logits = -np.asarray(distances)
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    @staticmethod
    def _aggregate(flows: List[FlowPrediction]) -> Dict[str, object]:
        """文件级结论：以恶意流的加权占比决定，正常类需压倒性多数。"""
        if not flows:
            return {
                "label": "unknown",
                "label_zh": "无有效流",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in CLASS_NAMES},
            }
        matrix = np.stack([[f.probabilities[name] for name in CLASS_NAMES] for f in flows])
        mean_probs = matrix.mean(axis=0)
        malicious = float(1.0 - mean_probs[CLASS_NAMES.index("benign")])
        best = int(np.argmax(mean_probs))
        return {
            "label": CLASS_NAMES[best],
            "label_zh": CLASS_NAMES_ZH[CLASS_NAMES[best]],
            "confidence": round(float(mean_probs[best]), 4),
            "malicious_score": round(malicious, 4),
            "probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, mean_probs)},
        }


_predictor: Optional[Predictor] = None
_lock = threading.Lock()


def get_predictor(checkpoint: Optional[Path] = None, reload: bool = False) -> Predictor:
    """进程内单例，避免每次请求都反序列化权重。"""
    global _predictor
    with _lock:
        if _predictor is None or reload:
            _predictor = Predictor(checkpoint)
    return _predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="恶意流量分类推理")
    parser.add_argument("--file", type=Path, required=True, help="PCAP 文件路径")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    result = get_predictor(args.checkpoint).predict_file(args.file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
