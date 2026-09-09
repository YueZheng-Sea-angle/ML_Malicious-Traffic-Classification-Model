"""DataCon 特征画像与选择报告（研究管线 probe/profile/selector 的落盘实现）。

用法示例：
    python -m ml.research.profile --cache artifacts/research/part1_sample_tools11.npz
    python -m ml.research.profile --rebuild --real-per-class 4   # 重建缓存后再分析

产物写入 artifacts/research/：
    probe_<tag>.json         文件/流/包数统计画像
    profile_<tag>.json       40 维统计特征按类均值与标准差
    selector_<tag>.json      FeatureSelector 贡献报告（互信息/方差/树加权融合）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.config import PROJECT_ROOT
from ml.data.dataset import SampleBatch
from ml.features.extractor import STAT_FEATURE_NAMES
from ml.features.selection import FeatureSelector
from ml.research.datacon import DEFAULT_RESEARCH_DIR, DataConLoader, flow_file_key

_N_SELECT = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataCon 特征画像与选择报告")
    p.add_argument("--rebuild", action="store_true", help="忽略缓存重新解析")
    p.add_argument("--real-per-class", type=int, default=0, help="每类补充 real_data 文件数")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_batch(args) -> SampleBatch:
    tag = f"sample_rpc{args.real_per_class}" if args.real_per_class else "sample"
    cache = args.cache or Path(args.out_dir) / f"part1_{tag}_tools11.npz"
    if args.rebuild or not cache.exists():
        batch = DataConLoader().build(include_sample=True, real_per_class=args.real_per_class,
                                      cache=None if args.rebuild else cache, seed=args.seed)
        if not args.rebuild:
            cache.parent.mkdir(parents=True, exist_ok=True)
            from ml.data.dataset import save_dataset
            save_dataset(batch, cache)
    else:
        from ml.data.dataset import load_dataset
        batch = load_dataset(cache)
        print(f"[profile] 从缓存加载 {cache}（{len(batch)} 样本）")
    return batch


def _median(x: np.ndarray):
    return float(np.median(x)) if len(x) else None


def _probe(batch: SampleBatch) -> dict:
    groups = {}
    for i, fid in enumerate(batch.flow_ids):
        groups.setdefault(flow_file_key(fid), []).append(i)
    by_class: dict = {}
    for key, idxs in groups.items():
        c = int(batch.labels[idxs[0]])
        by_class.setdefault(c, []).append(np.asarray(idxs))
    names = list(batch.class_names)
    pkt_idx, dur_idx = 0, 2  # STAT_FEATURE_NAMES 中的 pkt_count / duration
    out = {}
    for c, idxs in by_class.items():
        idx = np.concatenate(idxs)
        per_file = [len(x) for x in idxs]
        out[str(names[c])] = {
            "files": len(idxs),
            "flows": int(len(idx)),
            "flows_per_file": {"min": min(per_file), "median": int(np.median(per_file)),
                               "max": max(per_file)},
            "pkt_per_flow_median": int(np.median(batch.stats[idx, pkt_idx])),
            "flow_duration_median_s": round(float(np.median(batch.stats[idx, dur_idx])), 3),
        }
    return {"n_class": len(by_class), "total_flows": int(len(batch)), "by_class": out}


def _profile_features(batch: SampleBatch) -> dict:
    names = list(batch.class_names)
    out = {}
    for c, name in enumerate(names):
        m = batch.labels == c
        if not m.any():
            continue
        x = batch.stats[m]
        out[name] = {
            "n": int(m.sum()),
            "features": {
                fname: {"mean": round(float(x[:, i].mean()), 4),
                        "std": round(float(x[:, i].std()), 4)}
                for i, fname in enumerate(STAT_FEATURE_NAMES)
            },
        }
    return out


def _selector_report(batch: SampleBatch) -> dict:
    x = np.nan_to_num(batch.stats.astype(np.float64))
    selector = FeatureSelector().fit(x, batch.labels)
    report = selector.to_dict()
    print("\n[selector] 贡献 Top 10：")
    for s in selector.top(10):
        print(f"  {s.name:<26} contrib={s.contribution:.4f}  "
              f"mi={s.detail['mutual_info']:.4f} var={s.detail['variance']:.4f} "
              f"tree={s.detail['tree']:.4f}")
    print(f"[selector] 自适应保留 {len(selector.selected_)}/{x.shape[1]} 维"
          f"（coverage={selector.coverage}）")
    return report


def main() -> None:
    args = parse_args()
    batch = load_batch(args)
    tag = f"sample_rpc{args.real_per_class}" if args.real_per_class else "sample"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        f"probe_{tag}.json": _probe(batch),
        f"profile_{tag}.json": _profile_features(batch),
        f"selector_{tag}.json": _selector_report(batch),
    }
    for fname, payload in artifacts.items():
        (args.out_dir / fname).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[profile] 已写入 {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
