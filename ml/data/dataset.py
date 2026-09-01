"""数据集：真实 PCAP 目录构建 + 合成样本生成。

正式数据集到位前，用 make_synthetic_dataset 打通「训练 -> 保存 -> 推理 -> 前端展示」
全链路；数据集到位后改用 build_dataset_from_pcaps，其余代码无需改动。

真实数据集期望的目录结构：
    data/raw/<类别名>/xxx.pcap        类别名取自 ml.config.CLASS_NAMES
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ml.config import BYTE_SEQ_LEN, CLASS_NAMES, PKT_SEQ_LEN
from ml.features.extractor import STAT_DIM, Flow, Packet, flow_to_sample, pcap_to_samples


@dataclass
class SampleBatch:
    """一批样本的三路输入与标签。"""

    stats: np.ndarray   # (N, STAT_DIM) float32
    pkt: np.ndarray     # (N, PKT_SEQ_LEN) float32
    byte: np.ndarray    # (N, BYTE_SEQ_LEN) int64
    labels: np.ndarray  # (N,) int64
    flow_ids: List[str]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def subset(self, index: np.ndarray) -> "SampleBatch":
        return SampleBatch(
            stats=self.stats[index],
            pkt=self.pkt[index],
            byte=self.byte[index],
            labels=self.labels[index],
            flow_ids=[self.flow_ids[i] for i in index],
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
        raise FileNotFoundError(f"{root} 下未找到任何 PCAP，请检查目录结构或改用合成数据集")
    return _stack(stats, pkts, bytes_, labels, flow_ids)


def make_synthetic_dataset(samples_per_class: int = 150, seed: int = 42) -> SampleBatch:
    """按类别先验生成伪流，再走真实特征提取管线。"""
    rng = np.random.default_rng(seed)
    stats, pkts, bytes_, labels, flow_ids = [], [], [], [], []
    for label, class_name in enumerate(CLASS_NAMES):
        for i in range(samples_per_class):
            sample = flow_to_sample(_synthesize_class_flow(label, rng))
            # 流间关联特征在合成数据下按类别先验给定
            sample.stats[-3:] = _CLASS_PROFILES[class_name]["cross_flow"] * (
                1.0 + rng.normal(0, 0.08, size=3)
            )
            stats.append(sample.stats)
            pkts.append(sample.pkt_seq)
            bytes_.append(sample.byte_seq)
            labels.append(label)
            flow_ids.append(f"synthetic/{class_name}/{i:04d}")
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
    )


def load_dataset(path: Path) -> SampleBatch:
    data = np.load(Path(path), allow_pickle=True)
    return SampleBatch(
        stats=data["stats"],
        pkt=data["pkt"],
        byte=data["byte"],
        labels=data["labels"],
        flow_ids=[str(x) for x in data["flow_ids"]],
    )


# --------------------------------------------------------------------------- #
# 合成流的类别先验
# 字段含义：包长基线 / 包长抖动 / 客户端方向概率 / 平均包间隔 / 载荷可打印比例 / 包数范围
# --------------------------------------------------------------------------- #
_CLASS_PROFILES = {
    "benign": dict(base_len=780, jitter=0.40, fwd_p=0.45, iat=0.030, printable=0.10,
                   pkts=(25, 70), cross_flow=np.array([2.0, 2.0, 0.4])),
    "botnet": dict(base_len=210, jitter=0.12, fwd_p=0.62, iat=0.900, printable=0.05,
                   pkts=(8, 24), cross_flow=np.array([9.0, 3.0, 0.9])),
    "ransomware": dict(base_len=1180, jitter=0.20, fwd_p=0.80, iat=0.012, printable=0.02,
                       pkts=(40, 90), cross_flow=np.array([3.0, 5.0, 0.7])),
    "trojan": dict(base_len=430, jitter=0.55, fwd_p=0.55, iat=0.180, printable=0.20,
                   pkts=(15, 45), cross_flow=np.array([5.0, 2.0, 0.5])),
    "cryptomining": dict(base_len=160, jitter=0.08, fwd_p=0.50, iat=0.250, printable=0.35,
                         pkts=(30, 80), cross_flow=np.array([1.0, 1.0, 1.0])),
    "ddos": dict(base_len=90, jitter=0.05, fwd_p=0.95, iat=0.002, printable=0.01,
                 pkts=(60, 120), cross_flow=np.array([14.0, 12.0, 1.0])),
}


def _synthesize_class_flow(label: int, rng: np.random.Generator) -> Flow:
    profile = _CLASS_PROFILES[CLASS_NAMES[label]]
    packet_count = int(rng.integers(*profile["pkts"]))
    timestamp = 0.0
    packets: List[Packet] = []
    for i in range(packet_count):
        timestamp += float(abs(rng.exponential(profile["iat"]))) + 1e-5
        direction = 1 if rng.random() < profile["fwd_p"] else -1
        length = int(
            np.clip(rng.normal(profile["base_len"], profile["base_len"] * profile["jitter"]), 60, 1514)
        )
        packets.append(Packet(timestamp, length, direction, _synthesize_payload(rng, profile, i)))
    return Flow(
        src=f"10.0.0.{int(rng.integers(2, 250))}",
        dst=f"172.16.{int(rng.integers(0, 250))}.{int(rng.integers(2, 250))}",
        sport=int(rng.integers(1024, 65535)),
        dport=443,
        proto="TCP",
        packets=packets,
    )


def _synthesize_payload(rng: np.random.Generator, profile: dict, index: int) -> bytes:
    size = 48
    printable_count = int(size * profile["printable"])
    body = list(rng.integers(0, 256, size=size - printable_count))
    body += list(rng.integers(32, 127, size=printable_count))
    rng.shuffle(body)
    if index < 3:
        head = [0x16, 0x03, 0x03]   # 握手记录
    else:
        head = [0x17, 0x03, 0x03]   # 应用数据记录
    return bytes(head + [int(b) for b in body[3:]])


def _stack(stats, pkts, bytes_, labels, flow_ids) -> SampleBatch:
    return SampleBatch(
        stats=np.asarray(stats, dtype=np.float32).reshape(-1, STAT_DIM),
        pkt=np.asarray(pkts, dtype=np.float32).reshape(-1, PKT_SEQ_LEN),
        byte=np.asarray(bytes_, dtype=np.int64).reshape(-1, BYTE_SEQ_LEN),
        labels=np.asarray(labels, dtype=np.int64),
        flow_ids=list(flow_ids),
    )
