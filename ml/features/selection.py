"""特征贡献评估与自适应选择。

对应需求「特征贡献评估与自适应选择机制；动态筛选与融合关键特征」：
    1. 三种打分互补——互信息（非线性相关）、方差（区分度）、树模型重要性（交互作用）
    2. 归一化加权融合为统一贡献度
    3. 自适应阈值：按累计贡献覆盖率 coverage 取前 k 维，而非人工定死 k

sklearn 缺失时退化为「|皮尔逊相关| + 方差」的轻量打分，接口保持一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ml.features.extractor import STAT_FEATURE_NAMES

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    SKLEARN_AVAILABLE = False


@dataclass
class FeatureScore:
    name: str
    index: int
    contribution: float
    detail: Dict[str, float]


class FeatureSelector:
    """统计特征的贡献评估器 + 掩码生成器。"""

    def __init__(
        self,
        feature_names: Optional[Sequence[str]] = None,
        coverage: float = 0.95,
        min_features: int = 8,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.feature_names = list(feature_names or STAT_FEATURE_NAMES)
        self.coverage = coverage
        self.min_features = min_features
        self.weights = weights or {"mutual_info": 0.4, "variance": 0.2, "tree": 0.4}
        self.scores_: List[FeatureScore] = []
        self.selected_: List[int] = []

    # ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        y = np.asarray(y).ravel()
        detail = {
            "mutual_info": self._mutual_info(x, y),
            "variance": self._variance(x),
            "tree": self._tree_importance(x, y),
        }
        fused = sum(self.weights[k] * v for k, v in detail.items())
        fused = fused / (fused.sum() + 1e-12)

        self.scores_ = [
            FeatureScore(
                name=self.feature_names[i],
                index=i,
                contribution=float(fused[i]),
                detail={k: float(v[i]) for k, v in detail.items()},
            )
            for i in range(x.shape[1])
        ]
        self.scores_.sort(key=lambda s: s.contribution, reverse=True)
        self.selected_ = self._adaptive_select(fused)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.selected_:
            return np.asarray(x)
        return np.asarray(x)[:, self.selected_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(x, y).transform(x)

    def mask(self, dim: Optional[int] = None) -> np.ndarray:
        """返回 0/1 掩码，供模型做软融合（保持输入维度不变）。"""
        dim = dim or len(self.feature_names)
        m = np.zeros(dim, dtype=np.float32)
        m[self.selected_] = 1.0
        return m

    def top(self, k: int = 10) -> List[FeatureScore]:
        return self.scores_[:k]

    def to_dict(self) -> Dict[str, object]:
        return {
            "coverage": self.coverage,
            "selected": self.selected_,
            "selected_names": [self.feature_names[i] for i in self.selected_],
            "scores": [
                {"name": s.name, "index": s.index, "contribution": s.contribution, "detail": s.detail}
                for s in self.scores_
            ],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    def _adaptive_select(self, fused: np.ndarray) -> List[int]:
        order = np.argsort(fused)[::-1]
        cumulative = np.cumsum(fused[order])
        k = int(np.searchsorted(cumulative, self.coverage) + 1)
        k = max(self.min_features, min(k, len(order)))
        return sorted(int(i) for i in order[:k])

    def _mutual_info(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if SKLEARN_AVAILABLE:
            try:
                return _normalize(mutual_info_classif(x, y, random_state=0))
            except Exception:
                pass
        return _normalize(_abs_correlation(x, y))

    @staticmethod
    def _variance(x: np.ndarray) -> np.ndarray:
        scale = x.std(axis=0) + 1e-9
        normalized = (x - x.mean(axis=0)) / scale
        return _normalize(normalized.var(axis=0) * np.log1p(scale))

    def _tree_importance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if SKLEARN_AVAILABLE and len(np.unique(y)) > 1:
            try:
                forest = RandomForestClassifier(
                    n_estimators=120, random_state=0, n_jobs=-1, max_depth=12
                )
                forest.fit(x, y)
                return _normalize(forest.feature_importances_)
            except Exception:
                pass
        return _normalize(_abs_correlation(x, y))


def _abs_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64)
    y_centered = y - y.mean()
    x_centered = x - x.mean(axis=0)
    denom = (np.linalg.norm(x_centered, axis=0) * np.linalg.norm(y_centered)) + 1e-12
    return np.abs(x_centered.T @ y_centered) / denom


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    total = values.sum()
    if total <= 0:
        return np.full_like(values, 1.0 / len(values))
    return values / total
