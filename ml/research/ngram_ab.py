"""DataCon T2（隧道/正常二元）n-gram 特征增强 A/B 对照实验。

    A 组（基线）：仅 40 维统计特征，不使用 n-gram；
    B 组（增强）：40 维统计特征 + 载荷字节 n-gram（n=1..max_n，chi2 top-k + TF-IDF）。

对照实验设计（公平性原则）：
    * 同一文件级分层划分（防同源泄漏，见 ml/research/split.py 语义），单次 seed 复现；
    * 统计特征标准化参数与 n-gram 词表/TF-IDF 一律只取自训练集，验证集仅变换；
    * A/B 组分别在同一批分类器（LR、RF）上用同一指标（accuracy / macro-F1 / 逐类召回）评估。

用法示例：
    python -m ml.research.ngram_ab --real-per-class 3
    python -m ml.research.ngram_ab --real-per-class 0 --ngram-max-n 4 --ngram-topk 500
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from ml.features.extractor import (
    STAT_DIM,
    STAT_FEATURE_NAMES,
    _augment_cross_flow,
    extract_flows,
    flow_to_sample,
)
from ml.research.datacon import (
    DEFAULT_RESEARCH_DIR,
    NORMAL_NEG,
    TOOLS_11,
    TUNNEL_POS,
    DataConLoader,
)
from ml.research.ngram import ByteNgramVectorizer, payload_blob

CLASS_NAMES = ["normal", "tunnel"]

MODELS = {
    "LR": lambda seed: LogisticRegression(
        C=1.0, max_iter=3000, class_weight="balanced", solver="lbfgs"
    ),
    "RF": lambda seed: RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    ),
}


# --------------------------------------------------------------------------- #
# 数据装载（与 datacon.build 同构，但并行保留 n-gram 原始载荷）
# --------------------------------------------------------------------------- #
def build_t2_rows(
    loader: DataConLoader, real_per_class: int, max_flows: int, seed: int
) -> Tuple[List[np.ndarray], List[bytes], List[int], List[str]]:
    files: List[Tuple[Path, int]] = loader.sample_items()
    if real_per_class:
        try:
            files.extend(loader.real_items(per_class=real_per_class, seed=seed))
        except FileNotFoundError as exc:
            print(f"[T2装载] {exc}，忽略 --real-per-class={real_per_class}，仅使用 sample")

    stats, blobs, labels, flow_ids = [], [], [], []
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
            blobs.append(payload_blob(flow))
            labels.append(c)
            flow_ids.append(f"{p.stem}::{sample.flow_id}")
    print(f"[T2装载] 文件 {n_file}（跳过 {fail}）→ 流样本 {len(labels)}")
    return stats, blobs, labels, flow_ids


def agent_filter(
    stats: Sequence[np.ndarray], blobs: Sequence[bytes],
    labels: Sequence[int], flow_ids: Sequence[str],
) -> Tuple[List[np.ndarray], List[bytes], List[int], List[str]]:
    keep_s, keep_b, keep_y, keep_f = [], [], [], []
    for s, b, lab, fid in zip(stats, blobs, labels, flow_ids):
        name = TOOLS_11[int(lab)]
        if name in NORMAL_NEG:
            keep_s.append(s); keep_b.append(b); keep_y.append(0); keep_f.append(fid)
        elif name in TUNNEL_POS:
            keep_s.append(s); keep_b.append(b); keep_y.append(1); keep_f.append(fid)
    return keep_s, keep_b, keep_y, keep_f


# --------------------------------------------------------------------------- #
# 文件级分层划分（与 split_by_file 语义一致，返回行下标）
# --------------------------------------------------------------------------- #
def split_rows(
    keys: Sequence[str], y: Sequence[int], val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    by_label: Dict[int, List[Tuple[str, List[int]]]] = {}
    for key, idxs in groups.items():
        by_label.setdefault(int(y[idxs[0]]), []).append((key, idxs))

    train_idx, val_idx, singleton = [], [], 0
    for label, entries in by_label.items():
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
        print(f"[split_rows] {singleton} 类因仅单文件全部进入训练（验证集缺该类）")
    train = np.asarray(sorted(train_idx), dtype=np.int64)
    val = np.asarray(sorted(val_idx), dtype=np.int64)
    print(f"[split_rows] 训练 {len(train)} / 验证 {len(val)}（文件级不重叠）")
    return train, val


# --------------------------------------------------------------------------- #
def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    recall = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "per_class_recall": {
            CLASS_NAMES[i]: round(float(r), 5) for i, r in enumerate(recall)
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataCon T2 n-gram A/B 对照实验")
    p.add_argument("--real-per-class", type=int, default=3,
                   help="每类补充 real_data 文件数(0=只用 sample)")
    p.add_argument("--max-flows", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ngram-max-n", type=int, default=4)
    p.add_argument("--ngram-min-df", type=int, default=2)
    p.add_argument("--ngram-topk", type=int, default=500)
    p.add_argument("--ngram-max-packets", type=int, default=16)
    p.add_argument("--ngram-max-bytes", type=int, default=1024)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loader = DataConLoader()
    stats, blobs, labels, flow_ids = build_t2_rows(
        loader, args.real_per_class, args.max_flows, args.seed
    )
    stats, blobs, labels, flow_ids = agent_filter(stats, blobs, labels, flow_ids)
    if len(labels) < 20:
        raise SystemExit("保留样本过少，请增大 --real-per-class")
    class_cnt = Counter(labels)
    print(f"[T2口径] normal={class_cnt[0]} tunnel={class_cnt[1]}")

    train, val = split_rows(
        [fid.split("::", 1)[0] for fid in flow_ids], labels, args.val_ratio, args.seed
    )
    y_tr = np.asarray([labels[i] for i in train], dtype=np.int64)
    y_va = np.asarray([labels[i] for i in val], dtype=np.int64)

    S_tr = np.vstack([stats[i] for i in train]).astype(np.float64)
    S_va = np.vstack([stats[i] for i in val]).astype(np.float64)
    scaler = StandardScaler().fit(S_tr)
    XA_tr = scaler.transform(S_tr)
    XA_va = scaler.transform(S_va)

    blobs_tr = [blobs[i] for i in train]
    blobs_va = [blobs[i] for i in val]
    vec = ByteNgramVectorizer(
        max_n=args.ngram_max_n, min_df=args.ngram_min_df, top_k=args.ngram_topk,
        max_packets=args.ngram_max_packets, max_bytes=args.ngram_max_bytes,
    ).fit(blobs_tr, y_tr)
    ng_tr = vec.transform(blobs_tr)
    ng_va = vec.transform(blobs_va)
    print(f"[n-gram] 词表 {len(vec.vocab_)} 个（n=1..{args.ngram_max_n}，"
          f"min_df={args.ngram_min_df}，chi2 top-{args.ngram_topk}）")

    XB_tr = sparse.hstack([sparse.csr_matrix(XA_tr), ng_tr]).tocsr()
    XB_va = sparse.hstack([sparse.csr_matrix(XA_va), ng_va]).tocsr()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    print(f"\n{'特征组':<10}{'模型':<5}{'accuracy':>10}{'macro-F1':>10}  per-class recall")
    for feats, (X_tr, X_va) in {
        "A 统计40": (XA_tr, XA_va),
        "B 统计40+ngram": (XB_tr, XB_va),
    }.items():
        results[feats] = {}
        for name, build in MODELS.items():
            clf = build(args.seed)
            clf.fit(X_tr, y_tr)
            metric = evaluate(y_va, clf.predict(X_va))
            results[feats][name] = metric
            rec = "  ".join(f"{CLASS_NAMES[i]}={metric['per_class_recall'][CLASS_NAMES[i]]}"
                            for i in range(2))
            print(f"{feats:<10}{name:<5}{metric['accuracy']:>10.4f}"
                  f"{metric['macro_f1']:>10.4f}  {rec}")

    report = {
        "task": "agent2",
        "description": "A=40维统计(基线) / B=40维统计+载荷字节n-gram；文件级划分；LR/RF",
        "class_names": CLASS_NAMES,
        "params": {
            "real_per_class": args.real_per_class, "max_flows": args.max_flows,
            "val_ratio": args.val_ratio, "seed": args.seed,
            "ngram_max_n": args.ngram_max_n, "ngram_min_df": args.ngram_min_df,
            "ngram_topk": args.ngram_topk, "ngram_max_packets": args.ngram_max_packets,
            "ngram_max_bytes": args.ngram_max_bytes,
        },
        "n_flows": {"train": int(len(train)), "val": int(len(val))},
        "class_dist_val": {CLASS_NAMES[int(k)]: int(v) for k, v in Counter(y_va).items()},
        "stat_dim": STAT_DIM,
        "stat_feature_names": list(STAT_FEATURE_NAMES),
        "ngram_vocab_size": len(vec.vocab_),
        "results": results,
    }
    out = args.out or Path(DEFAULT_RESEARCH_DIR) / "ngram_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[实验] 报告已写入 {out}")


if __name__ == "__main__":
    main()
