"""文件级分层划分：以 pcap 文件为最小单元切分训练/验证集。

同一 pcap 内切出的流同标签且同源，若按流随机划分会造成同源泄漏、指标虚高；
本模块保证验证集与训练集在「文件」粒度上不重叠（对应《特征方案与详细设计.md》决策 D3）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ml.data.dataset import SampleBatch
from ml.research.datacon import flow_file_key


def file_groups(batch: SampleBatch) -> Dict[str, List[int]]:
    """按文件分组样本下标，返回 {文件key: [样本下标,...]}。"""
    groups: Dict[str, List[int]] = {}
    for i, fid in enumerate(batch.flow_ids):
        groups.setdefault(flow_file_key(fid), []).append(i)
    return groups


def split_by_file(batch: SampleBatch, val_ratio: float = 0.2,
                  seed: int = 0) -> Tuple[SampleBatch, SampleBatch]:
    """按文件分层划分。单文件类别（如 sample 中每类仅 1 个文件的类）全部进训练并告警。

    返回 (训练集, 验证集)。"""
    rng = np.random.default_rng(seed)
    groups = file_groups(batch)

    by_label: Dict[int, List[Tuple[str, List[int]]]] = {}
    for key, idxs in groups.items():
        by_label.setdefault(int(batch.labels[idxs[0]]), []).append((key, idxs))

    train_idx, val_idx, singleton = [], [], 0
    for label, entries in by_label.items():
        rng.shuffle(entries)
        if len(entries) < 2:  # 该类只有一个文件：全部放训练，验证集不评估该类
            for _, idxs in entries:
                train_idx.extend(idxs)
            singleton += 1
            continue
        n_val = max(1, int(round(len(entries) * val_ratio)))
        n_val = min(n_val, len(entries) - 1)  # 保证训练侧至少留一个文件
        for _, idxs in entries[:n_val]:
            val_idx.extend(idxs)
        for _, idxs in entries[n_val:]:
            train_idx.extend(idxs)

    if singleton:
        print(f"[split_by_file] {singleton} 个类别因只有单文件而全部进入训练集"
              f"（验证集将缺失该类，属预期小样本场景）")

    train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
    val_idx = np.asarray(sorted(val_idx), dtype=np.int64)
    print(f"[split_by_file] 训练 {len(train_idx)} / 验证 {len(val_idx)}，"
          f"文件级不重叠，seed={seed}")
    return batch.subset(train_idx), batch.subset(val_idx)
