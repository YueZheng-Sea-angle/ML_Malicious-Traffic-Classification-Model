"""推理入口：PCAP 文件 -> 流级与文件级分类结果。

产品分类口径为 DataCon T1（11 类加密代理/隧道工具形态识别），类别顺序与
checkpoint 中保存的 class_names 一致，中文表取自 ml.config。

模型分两组（均为 T1，基座为 LightGBM 分类器）：
    A 组（基线）：特征 = 40 维统计（malflow_datacon_tools.pt，默认激活）
    B 组（增强）：特征 = 40 维统计 + 载荷字节 n-gram（malflow_datacon_tools_ngram.pt）

运行模式：
    model        已训练权重可用（torch 文件内嵌模型，含 PyTorch 或 LightGBM）
    unavailable  权重缺失或 torch 未安装，返回占位结果

两种模式返回结构一致，前端与接口契约不受影响；结果中的 ``mode`` 字段用于界面提示。

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
from ml.features.extractor import (
    STAT_FEATURE_NAMES,
    FlowSample,
    extract_flows,
    pcap_to_samples,
)

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    TORCH_AVAILABLE = False

LABEL_UNKNOWN = "unknown"
LABEL_UNKNOWN_ZH = "无法分类（无可用权重）"


def _payload_blob(flow, max_packets: int, max_bytes: int) -> bytes:
    parts = [p.payload for p in flow.packets[:max_packets]]
    return b"".join(parts)[:max_bytes]


def _blob_from_sample(sample: FlowSample) -> bytes:
    """退化分支：无法重解析流时，用样本内置的前 256 字节重建 n-gram 文本。"""
    seq = sample.byte_seq
    raw = bytes(int(b) - 1 for b in seq if int(b) > 0)
    return raw[:256]


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
        self.mode = "unavailable"
        self.kind: Optional[str] = None   # torch / lightgbm
        self.model = None
        self.stat_mean: Optional[np.ndarray] = None
        self.stat_std: Optional[np.ndarray] = None
        self.selected_features: List[str] = []
        self.metrics: Dict[str, object] = {}
        self.class_names: List[str] = list(CLASS_NAMES)          # 随 checkpoint 覆盖
        self.class_names_zh: Dict[str, str] = dict(CLASS_NAMES_ZH)
        # n-gram（B 组）相关，随 checkpoint 覆盖
        self.ngram_dim = 0
        self.ngram_max_packets = 16
        self.ngram_max_bytes = 1024
        self.ngram_vectorizer = None
        self.ngram_scaler = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not (TORCH_AVAILABLE and self.checkpoint_path.exists()):
            return
        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            names = list(payload.get("class_names", CLASS_NAMES))
            self.stat_mean = np.asarray(payload["stat_mean"], dtype=np.float32)
            self.stat_std = np.asarray(payload["stat_std"], dtype=np.float32)
            self.selected_features = list(payload.get("selected_features", []))
            self.metrics = payload.get("metrics", {})
            self.ngram_dim = int(payload.get("ngram_dim", 0) or 0)
            self.ngram_max_packets = int(payload.get("ngram_max_packets", 16))
            self.ngram_max_bytes = int(payload.get("ngram_max_bytes", 1024))
            self.ngram_vectorizer = payload.get("ngram_vectorizer")
            self.ngram_scaler = payload.get("ngram_scaler")

            if payload.get("kind") == "lightgbm":
                import lightgbm as lgb

                booster = lgb.Booster(model_str=payload["lightgbm_model_str"])
                self.model = booster
                self.kind = "lightgbm"
            else:
                from ml.models.cnn_bilstm import build_model

                model = build_model(len(names), ngram_dim=self.ngram_dim)
                model.load_state_dict(payload["state_dict"])
                model.eval()
                self.model = model
                self.kind = "torch"

            self.class_names = names
            self.class_names_zh = {n: CLASS_NAMES_ZH.get(n, n) for n in names}
            self.mode = "model"
        except Exception as exc:  # 权重损坏不应导致服务不可用
            print(f"[predict] 加载权重失败：{exc}")

    def info(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "kind": self.kind,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "torch_available": TORCH_AVAILABLE,
            "class_names": self.class_names,
            "ngram_dim": self.ngram_dim,
            "selected_features": self.selected_features,
            "metrics": self.metrics,
        }

    # ------------------------------------------------------------------ #
    def _ngram_active(self) -> bool:
        return (
            self.mode == "model"
            and self.ngram_dim > 0
            and self.ngram_vectorizer is not None
            and self.ngram_scaler is not None
        )

    def _ngram_rows(self, path: Path, max_flows: int, n_samples: int) -> Optional[np.ndarray]:
        """为 B 组模型构造逐流 n-gram 特征（已标准化）。"""
        try:
            flows = extract_flows(path)[:max_flows]
            if len(flows) != n_samples:  # 对齐失败退化为样本内置字节
                flows = []
        except Exception:
            flows = []
        vectorizer, scaler = self.ngram_vectorizer, self.ngram_scaler
        if flows:
            blobs = [_payload_blob(f, self.ngram_max_packets, self.ngram_max_bytes) for f in flows]
        else:
            raise RuntimeError("无法为 n-gram 模型重建流载荷，请检查 pcap 解析")
        matrix = vectorizer.transform(blobs)
        return scaler.transform(matrix.toarray()).astype(np.float32)

    def predict_file(self, path: Path, max_flows: int = 64) -> Dict[str, object]:
        path = Path(path)
        samples = pcap_to_samples(path, max_flows=max_flows)
        ngram = self._ngram_rows(path, max_flows, len(samples)) if self._ngram_active() else None
        flow_results = []
        for i, sample in enumerate(samples):
            row = ngram[i] if ngram is not None else None
            flow_results.append(self.predict_sample(sample, row))
        if self.mode != "model":
            names = self.class_names
            return {
                "file": path.name,
                "mode": self.mode,
                "flow_count": len(flow_results),
                "flows": [f.to_dict() for f in flow_results],
                "label": LABEL_UNKNOWN,
                "label_zh": LABEL_UNKNOWN_ZH,
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in names},
                "top_features": self.explain(samples),
            }
        return {
            "file": path.name,
            "mode": self.mode,
            "flow_count": len(flow_results),
            "flows": [f.to_dict() for f in flow_results],
            **self._aggregate(flow_results),
            "top_features": self.explain(samples),
        }

    def predict_sample(self, sample: FlowSample,
                       ngram_row: Optional[np.ndarray] = None) -> FlowPrediction:
        names = self.class_names
        if self.mode != "model":
            return FlowPrediction(
                flow_id=sample.flow_id,
                label=LABEL_UNKNOWN,
                label_zh=LABEL_UNKNOWN_ZH,
                confidence=0.0,
                probabilities={name: 0.0 for name in names},
                meta=sample.meta,
            )
        probs = self._probs_model(sample, ngram_row)
        best = int(np.argmax(probs))
        return FlowPrediction(
            flow_id=sample.flow_id,
            label=names[best],
            label_zh=self.class_names_zh.get(names[best], names[best]),
            confidence=float(probs[best]),
            probabilities={name: float(p) for name, p in zip(names, probs)},
            meta=sample.meta,
        )

    def explain(self, samples: List[FlowSample], top_k: int = 6) -> List[Dict[str, object]]:
        """返回对本次判定影响最大的统计特征，供界面展示。"""
        if not samples:
            return []
        stats = np.stack([s.stats for s in samples])
        if self.kind == "torch" and self.model is not None:
            with torch.no_grad():
                gate = self.model.stat_encoder(
                    torch.from_numpy(self._standardize(stats)), None
                )[1].mean(dim=0).numpy()
            weights = gate
        else:
            # 树模型/未加载权重时以「偏离均值的程度」近似贡献
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

    def _probs_model(self, sample: FlowSample,
                     ngram_row: Optional[np.ndarray] = None) -> np.ndarray:
        row = self._standardize(sample.stats[None, :])
        if self.ngram_dim > 0:
            if ngram_row is None:
                raise ValueError("模型启用了 n-gram 输入，缺少 ngram 特征")
            row = np.concatenate([row, ngram_row[None, :]], axis=1)
        if self.kind == "lightgbm":
            proba = self.model.predict(row, num_iteration=None)
            return np.asarray(proba, dtype=np.float64)[0]
        stat_dim = row.shape[1] - (self.ngram_dim if self.ngram_dim > 0 else 0)
        with torch.no_grad():
            stats = torch.from_numpy(row[:, :stat_dim])
            pkt = torch.from_numpy(sample.pkt_seq[None, :])
            byte = torch.from_numpy(sample.byte_seq[None, :])
            inputs = [stats, pkt, byte]
            if self.ngram_dim > 0:
                inputs.append(torch.from_numpy(row[:, stat_dim:]))
            out = self.model(*inputs)
            return torch.softmax(out["logits"], dim=-1)[0].numpy()

    def _aggregate(self, flows: List[FlowPrediction]) -> Dict[str, object]:
        """文件级结论：以全部流平均概率聚合，取最大类作为文件级判定。"""
        names = list(self.class_names)
        if not flows:
            return {
                "label": LABEL_UNKNOWN,
                "label_zh": "无有效流",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in names},
            }
        matrix = np.stack([[f.probabilities[name] for name in names] for f in flows])
        mean_probs = matrix.mean(axis=0)
        best = int(np.argmax(mean_probs))
        return {
            "label": names[best],
            "label_zh": self.class_names_zh.get(names[best], names[best]),
            "confidence": round(float(mean_probs[best]), 4),
            "probabilities": {n: round(float(p), 4) for n, p in zip(names, mean_probs)},
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
    parser = argparse.ArgumentParser(description="加密代理/隧道工具分类推理")
    parser.add_argument("--file", type=Path, required=True, help="PCAP 文件路径")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    result = get_predictor(args.checkpoint).predict_file(args.file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
