"""数据集：按类别目录从真实 PCAP 构建样本矩阵（SampleBatch）。

产品分类口径为 DataCon T1（11 类加密代理/隧道工具），目录名取
ml.config.CLASS_NAMES；DataCon-ETA 原始布局（part1/sample、real_data +
part1_label.txt）由 ml/research/datacon.py 装载后转成同格式 SampleBatch。

真实数据集期望的目录结构：
    data/raw/<类别名>/xxx.pcap        类别名取自 ml.config.CLASS_NAMES
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ml.config import BYTE_SEQ_LEN, CLASS_NAMES, PKT_SEQ_LEN
from ml.features.extractor import STAT_DIM, pcap_to_samples


@dataclass
class SampleBatch:
    """一批样本的三路输入、标签与类别名（类别名随批携带，支持任意任务口径）。"""

    stats: np.ndarray   # (N, STAT_DIM) float32
    pkt: np.ndarray     # (N, PKT_SEQ_LEN) float32
    byte: np.ndarray    # (N, BYTE_SEQ_LEN) int64
    labels: np.ndarray  # (N,) int64
    flow_ids: List[str]
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def subset(self, index: np.ndarray) -> "SampleBatch":
        return SampleBatch(
            stats=self.stats[index],
            pkt=self.pkt[index],
            byte=self.byte[index],
            labels=self.labels[index],
            flow_ids=[self.flow_ids[i] for i in index],
            class_names=list(self.class_names),
        )


# --------------------------------------------------------------------------- #
def build_dataset_from_pcaps(root: Path, max_flows_per_file: int = 64) -> SampleBatch:
    root = Path(root)
    stats, pkts, bytes_, labels, flow_ids = [], [], [], [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for pcap in sorted(class_dir.glob("*.pcap*")):
            for sample in pcap_to_samples(pcap, max_flows=max_flows_per_file):
                stats.append(sample.stats)
                pkts.append(sample.pkt_seq)
                bytes_.append(sample.byte_seq)
                labels.append(label)
                flow_ids.append(f"{pcap.name}::{sample.flow_id}")
    if not labels:
        raise FileNotFoundError(f"{root} 下未找到任何 PCAP，请检查目录结构")
    return _stack(stats, pkts, bytes_, labels, flow_ids)


def split_dataset(
    batch: SampleBatch, val_ratio: float = 0.2, seed: int = 0
) -> Tuple[SampleBatch, SampleBatch]:
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(batch))
    cut = int(len(batch) * (1 - val_ratio))
    return batch.subset(index[:cut]), batch.subset(index[cut:])


def save_dataset(batch: SampleBatch, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        stats=batch.stats,
        pkt=batch.pkt,
        byte=batch.byte,
        labels=batch.labels,
        flow_ids=np.array(batch.flow_ids, dtype=object),
        class_names=np.array(batch.class_names, dtype=object),
    )


def load_dataset(path: Path) -> SampleBatch:
    data = np.load(Path(path), allow_pickle=True)
    class_names = [str(x) for x in data["class_names"]] if "class_names" in data else list(CLASS_NAMES)
    return SampleBatch(
        stats=data["stats"],
        pkt=data["pkt"],
        byte=data["byte"],
        labels=data["labels"],
        flow_ids=[str(x) for x in data["flow_ids"]],
        class_names=class_names,
    )


def _stack(stats, pkts, bytes_, labels, flow_ids, class_names=None) -> SampleBatch:
    return SampleBatch(
        stats=np.asarray(stats, dtype=np.float32).reshape(-1, STAT_DIM),
        pkt=np.asarray(pkts, dtype=np.float32).reshape(-1, PKT_SEQ_LEN),
        byte=np.asarray(bytes_, dtype=np.int64).reshape(-1, BYTE_SEQ_LEN),
        labels=np.asarray(labels, dtype=np.int64),
        flow_ids=list(flow_ids),
        class_names=list(class_names) if class_names else list(CLASS_NAMES),
    )
