"""DataCon T1 A/B 双模型训练：11 类代理工具形态识别（LightGBM 基座）。

    A 组（基线）：LightGBM，特征 = 标准化后的 40 维统计特征；
    B 组（增强）：LightGBM，特征 = 标准化后的 40 维统计特征 + 载荷字节 n-gram
                  （n=1..max_n，min_df 粗筛 + chi2 选 top-k + TF-IDF + 标准化，
                   词表只取自训练集）。

两条训练共享同一数据（文件级分层划分，防同源泄漏）与超参，产出两个 checkpoint：
    artifacts/models/malflow_datacon_tools.pt         A 组基线（默认激活）
    artifacts/models/malflow_datacon_tools_ngram.pt   B 组 n-gram 增强
checkpoint 内嵌 LightGBM Booster 文本（kind="lightgbm"），非 PyTorch 权重。
对比报告写入 artifacts/research/ab_tools_report.json。

用法：
    python -m ml.research.run_experiment --real-per-class 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from ml.config import MODEL_DIR
from ml.features.extractor import (
    _augment_cross_flow,
    extract_flows,
    flow_to_sample,
)
from ml.research.datacon import DEFAULT_RESEARCH_DIR, TOOLS_11, DataConLoader
from ml.research.ngram import ByteNgramVectorizer, payload_blob


# --------------------------------------------------------------------------- #
# 数据装载（样本 = 统计特征 + 原始载荷 blob，二者与流顺序对齐）
# --------------------------------------------------------------------------- #
def build_rows(loader: DataConLoader, real_per_class: int,
               max_flows: int, seed: int):
    files: List[Tuple[Path, int]] = loader.sample_items()
    if real_per_class:
        try:
            files.extend(loader.real_items(per_class=real_per_class, seed=seed))
        except FileNotFoundError as exc:
            print(f"[T1装载] {exc}，忽略 --real-per-class，仅使用 sample")

    stats, labels, flow_ids, blobs = [], [], [], []
    fail, n_file = 0, 0
    for p, c in files:
        try:
            flows = extract_flows(p, max_packets=20000)[:max_flows]
            if not flows:
                continue
            samples = [flow_to_sample(f) for f in flows]
            _augment_cross_flow(flows, samples)
        except Exception:
            fail += 1
            continue
        n_file += 1
        for sample, flow in zip(samples, flows):
            stats.append(sample.stats)
            labels.append(c)
            flow_ids.append(f"{p.stem}::{sample.flow_id}")
            blobs.append(payload_blob(flow))
    print(f"[T1装载] 文件 {n_file}（跳过 {fail}）→ 流样本 {len(labels)}")
    return (np.vstack(stats).astype(np.float64), np.asarray(labels, dtype=np.int64),
            flow_ids, blobs)


def split_by_file_rows(keys: Sequence[str], y: Sequence[int],
                       val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """文件级分层划分（语义同 ml/research/split.py），返回训练/验证行下标。"""
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    by_label: Dict[int, List[Tuple[str, List[int]]]] = {}
    for key, idxs in groups.items():
        by_label.setdefault(int(y[idxs[0]]), []).append((key, idxs))

    train_idx, val_idx, singleton = [], [], 0
    for entries in by_label.values():
        rng.shuffle(entries)
        if len(entries) < 2:
            for _, idxs in entries:
                train_idx.extend(idxs)
            singleton += 1
            continue
        n_val = min(max(1, int(round(len(entries) * val_ratio))), len(entries) - 1)
        for _, idxs in entries[:n_val]:
            val_idx.extend(idxs)
        for _, idxs in entries[n_val:]:
            train_idx.extend(idxs)
    if singleton:
        print(f"[split_rows] {singleton} 类因仅单文件全部进训练（验证集缺该类）")
    return np.asarray(sorted(train_idx), dtype=np.int64), np.asarray(sorted(val_idx), dtype=np.int64)


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(y.tolist())
    total = len(y)
    n = len(counts)
    weight = np.array([total / (n * counts[c]) for c in y])
    return weight / weight.mean()


def evaluate_clf(y_true: np.ndarray, proba: np.ndarray,
                 class_names: Sequence[str]) -> Dict[str, object]:
    y_pred = proba.argmax(axis=1)
    n = len(class_names)
    per_class, f1s = {}, []
    for c in range(n):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_names[c]] = round(recall, 4)
        f1s.append(f1)
    return {
        "accuracy": round(float((y_pred == y_true).mean()) if len(y_true) else 0.0, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class_recall": per_class,
    }


def fit_lightgbm(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, seed: int,
                 trees: int) -> Tuple[LGBMClassifier, Dict[str, object]]:
    clf = LGBMClassifier(
        n_estimators=trees, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=5, n_jobs=-1, random_state=seed, verbose=-1,
    )
    clf.fit(X_tr, y_tr, sample_weight=class_balanced_weights(y_tr))
    metrics = evaluate_clf(y_tr, clf.predict_proba(X_tr), TOOLS_11)
    return clf, metrics


def save_checkpoint(path: Path, clf: LGBMClassifier, mean: np.ndarray, std: np.ndarray,
                    metrics: Dict[str, object], extra: Optional[Dict[str, object]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "lightgbm",
        "lightgbm_model_str": clf.booster_.model_to_string(),
        "class_names": list(TOOLS_11),
        "task_scope": "tools11",
        "stat_mean": mean,
        "stat_std": std,
        "metrics": metrics,
        "selected_features": [],
        "version": "0.4.0",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataCon T1 A/B 双模型训练（LightGBM 基座）")
    p.add_argument("--real-per-class", type=int, default=5,
                   help="每类从 real_data 补抽文件数(0=只用 sample)")
    p.add_argument("--max-flows", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--trees", type=int, default=600, help="LightGBM 树数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ngram-max-n", type=int, default=4)
    p.add_argument("--ngram-min-df", type=int, default=2)
    p.add_argument("--ngram-topk", type=int, default=1000)
    p.add_argument("--ngram-max-packets", type=int, default=16)
    p.add_argument("--ngram-max-bytes", type=int, default=1024)
    p.add_argument("--out-baseline", type=Path, default=MODEL_DIR / "malflow_datacon_tools.pt")
    p.add_argument("--out-ngram", type=Path, default=MODEL_DIR / "malflow_datacon_tools_ngram.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats, labels, flow_ids, blobs = build_rows(
        DataConLoader(), args.real_per_class, args.max_flows, args.seed)
    print(f"[T1口径] 类别 {len(TOOLS_11)} | 样本 {len(labels)}")

    train, val = split_by_file_rows(
        [fid.split("::", 1)[0] for fid in flow_ids], labels, args.val_ratio, args.seed)
    y_tr, y_va = labels[train], labels[val]

    stat_scaler = StandardScaler().fit(stats[train])
    X_tr = stat_scaler.transform(stats[train])
    X_va = stat_scaler.transform(stats[val])

    report: Dict[str, object] = {
        "task": "tools11", "framework": "lightgbm", "class_names": list(TOOLS_11),
        "seed": args.seed, "real_per_class": args.real_per_class,
        "n_flows": {"train": int(len(train)), "val": int(len(val))},
    }

    # ---- A 组：LightGBM，仅 40 维统计 ---- #
    print("\n[训练 A 组 · LightGBM 基线（40 维统计）]")
    clf_a, metric_tr_a = fit_lightgbm(X_tr, y_tr, X_va, args.seed, args.trees)
    proba_a = clf_a.predict_proba(X_va)
    metric_a = evaluate_clf(y_va, proba_a, TOOLS_11)
    save_checkpoint(args.out_baseline, clf_a, stat_scaler.mean_, stat_scaler.scale_, metric_a)
    report["A_baseline"] = metric_a
    print(f"  train acc={metric_tr_a['accuracy']} f1={metric_tr_a['macro_f1']} | "
          f"val acc={metric_a['accuracy']} f1={metric_a['macro_f1']}")
    print(f"  -> 已保存 {args.out_baseline}")

    # ---- B 组：LightGBM，40 维统计 + 载荷字节 n-gram ---- #
    print("\n[训练 B 组 · LightGBM n-gram 增强（40 维统计 + 载荷字节 n-gram）]")
    vec = ByteNgramVectorizer(max_n=args.ngram_max_n, min_df=args.ngram_min_df,
                              top_k=args.ngram_topk, max_packets=args.ngram_max_packets,
                              max_bytes=args.ngram_max_bytes)
    ng_tr = vec.fit_transform([blobs[i] for i in train], y_tr)
    ng_va = vec.transform([blobs[i] for i in val])
    ng_scaler = StandardScaler().fit(ng_tr.toarray())
    Xb_tr = np.hstack([X_tr, ng_scaler.transform(ng_tr.toarray())])
    Xb_va = np.hstack([X_va, ng_scaler.transform(ng_va.toarray())])
    print(f"  n-gram 词表 {len(vec.vocab_)}（n=1..{args.ngram_max_n}，chi2 top-{args.ngram_topk}）")

    clf_b, metric_tr_b = fit_lightgbm(Xb_tr, y_tr, Xb_va, args.seed, args.trees)
    proba_b = clf_b.predict_proba(Xb_va)
    metric_b = evaluate_clf(y_va, proba_b, TOOLS_11)
    save_checkpoint(
        args.out_ngram, clf_b, stat_scaler.mean_, stat_scaler.scale_, metric_b,
        extra={"ngram_dim": int(len(vec.vocab_)),
               "ngram_max_packets": args.ngram_max_packets,
               "ngram_max_bytes": args.ngram_max_bytes,
               "ngram_vectorizer": vec,
               "ngram_scaler": ng_scaler},
    )
    report["B_ngram"] = metric_b
    report["ngram_vocab_size"] = len(vec.vocab_)
    print(f"  train acc={metric_tr_b['accuracy']} f1={metric_tr_b['macro_f1']} | "
          f"val acc={metric_b['accuracy']} f1={metric_b['macro_f1']}")
    print(f"  -> 已保存 {args.out_ngram}")

    out = Path(DEFAULT_RESEARCH_DIR) / "ab_tools_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[实验] 报告已写入 {out}")
    print(f"  A 基线:    val acc={metric_a['accuracy']} macro-F1={metric_a['macro_f1']}")
    print(f"  B n-gram:  val acc={metric_b['accuracy']} macro-F1={metric_b['macro_f1']}")


if __name__ == "__main__":
    main()
