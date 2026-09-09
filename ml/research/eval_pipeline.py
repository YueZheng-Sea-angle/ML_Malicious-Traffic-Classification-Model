"""模型性能测试流水线：real_data 随机文件级划分 -> 按 A/B 变体重训 LightGBM -> 测试集评估。

被「模型测试」页面调用。A/B 变体与产品口径一致：
    A：LightGBM，特征 = 标准化 40 维统计；
    B：LightGBM，特征 = 统计 + 载荷字节 n-gram（chi2 top-k + TF-IDF + 标准化）。
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from ml.features.extractor import _augment_cross_flow, extract_flows, flow_to_sample
from ml.research.datacon import TOOLS_11, DataConLoader
from ml.research.ngram import ByteNgramVectorizer, payload_blob

MAX_FLOWS_PER_FILE = 64


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def collect_real_files(per_class_files: int, seed: int) -> List[Tuple[Path, int]]:
    """从 real_data 每类随机抽 per_class_files 个文件（文件名 label 来自 part1_label.txt）。"""
    return DataConLoader().real_items(per_class=per_class_files, seed=seed)


def split_files_by_class(files: Sequence[Tuple[Path, int]],
                         train_ratio: float, seed: int):
    """文件级分层划分：每类内随机选 train_ratio 比例文件做训练，其余为测试。"""
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[Path, int]]] = {}
    for entry in files:
        by_label.setdefault(entry[1], []).append(entry)

    train_files, test_files = [], []
    for entries in by_label.values():
        rng.shuffle(entries)
        if len(entries) < 2:  # 单文件类全部入训练
            train_files.extend(entries)
            continue
        n_train = max(1, min(int(round(len(entries) * train_ratio)), len(entries) - 1))
        train_files.extend(entries[:n_train])
        test_files.extend(entries[n_train:])
    return train_files, test_files


def parse_rows(entries: Sequence[Tuple[Path, int]]):
    """解析文件列表为流级特征：统计特征、标签、flow_id、载荷 blob（顺序一致）。"""
    stats, labels, flow_ids, blobs = [], [], [], []
    skipped = 0
    for path, cls in entries:
        try:
            flows = extract_flows(path, max_packets=20000)[:MAX_FLOWS_PER_FILE]
            if not flows:
                continue
            samples = [flow_to_sample(f) for f in flows]
            _augment_cross_flow(flows, samples)
        except Exception:
            skipped += 1
            continue
        for sample, flow in zip(samples, flows):
            stats.append(sample.stats)
            labels.append(cls)
            flow_ids.append(f"{path.stem}::{sample.flow_id}")
            blobs.append(payload_blob(flow))
    if not labels:
        raise ValueError(f"{len(entries)} 个文件中无可用流（跳过 {skipped}）")
    return (np.vstack(stats).astype(np.float64), np.asarray(labels, dtype=np.int64),
            flow_ids, blobs, skipped)


# --------------------------------------------------------------------------- #
# 评估指标
# --------------------------------------------------------------------------- #
def flow_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, object]:
    y_pred = proba.argmax(axis=1)
    per_class, f1s = {}, []
    for c in range(len(TOOLS_11)):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[TOOLS_11[c]] = round(recall, 4)
        f1s.append(f1)
    return {
        "accuracy": round(float((y_pred == y_true).mean()) if len(y_true) else 0.0, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class_recall": per_class,
    }


def file_metrics(flow_ids: Sequence[str], y_true: np.ndarray,
                 proba: np.ndarray) -> Dict[str, float]:
    """文件级命中：同文件流取多数预测 vs 文件真实类别。"""
    pred = proba.argmax(axis=1)
    groups: Dict[str, Counter] = {}
    truth: Dict[str, int] = {}
    for fid, y, p in zip(flow_ids, y_true, pred):
        key = fid.split("::", 1)[0]
        groups.setdefault(key, Counter())[int(p)] += 1
        truth[key] = int(y)
    hits = sum(1 for key, c in groups.items() if c.most_common(1)[0][0] == truth.get(key, -1))
    return {"n_files": len(groups), "hit_rate": round(hits / len(groups), 4) if groups else 0.0}


def _class_balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(y.tolist())
    total = len(y)
    n = len(counts)
    weight = np.array([total / (n * counts[c]) for c in y])
    return weight / weight.mean()


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_eval(variant: str, train_ratio: float, per_class_files: int,
             seed: int = 42, trees: int = 300) -> Dict[str, object]:
    variant = variant.upper()
    if variant not in {"A", "B"}:
        raise ValueError(f"未知变体 {variant}，应为 A（统计基线）或 B（统计+n-gram）")
    train_ratio = float(np.clip(train_ratio, 0.05, 0.95))

    files = collect_real_files(per_class_files, seed)
    train_entries, test_entries = split_files_by_class(files, train_ratio, seed)
    if not test_entries:
        raise ValueError("按当前比例/每类文件数划分后测试集为空，请调大测试比例或每类文件数")

    S_tr, y_tr, _, blobs_tr, _ = parse_rows(train_entries)
    S_te, y_te, te_flow_ids, blobs_te, _ = parse_rows(test_entries)
    print(f"[eval] 变体 {variant} 训练文件 {len(train_entries)} -> {len(y_tr)} 流；"
          f"测试文件 {len(test_entries)} -> {len(y_te)} 流")

    scaler = StandardScaler().fit(S_tr)
    X_tr = scaler.transform(S_tr)
    X_te = scaler.transform(S_te)

    vocab_size = 0
    if variant == "B":
        vec = ByteNgramVectorizer(max_n=4, min_df=2, top_k=1000, max_packets=16, max_bytes=1024)
        ng_tr = vec.fit_transform(blobs_tr, y_tr)
        ng_te = vec.transform(blobs_te)
        ng_scaler = StandardScaler().fit(ng_tr.toarray())
        X_tr = np.hstack([X_tr, ng_scaler.transform(ng_tr.toarray())])
        X_te = np.hstack([X_te, ng_scaler.transform(ng_te.toarray())])
        vocab_size = len(vec.vocab_)

    clf = LGBMClassifier(
        n_estimators=trees, learning_rate=0.05, num_leaves=31, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, min_child_samples=5,
        n_jobs=-1, random_state=seed, verbose=-1,
    )
    clf.fit(X_tr, y_tr, sample_weight=_class_balanced_weights(y_tr))
    proba = clf.predict_proba(X_te)

    return {
        "variant": variant,
        "class_names": list(TOOLS_11),
        "n_train_files": len(train_entries),
        "n_test_files": len(test_entries),
        "n_train_flows": int(len(y_tr)),
        "n_test_flows": int(len(y_te)),
        "ngram_vocab_size": vocab_size,
        "test_metrics": flow_metrics(y_te, proba),
        "file_metrics": file_metrics(te_flow_ids, y_te, proba),
    }


if __name__ == "__main__":  # 自测：python -m ml.research.eval_pipeline
    print(run_eval(variant="A", train_ratio=0.3, per_class_files=2, trees=50))
