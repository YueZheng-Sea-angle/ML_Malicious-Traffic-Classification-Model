"""流量特征提取：PCAP -> 流 -> 三路特征（统计 / 包长方向序列 / 首部字节）。

对应需求「恶意流量特征的分析与表示」与「特征的增强与优化」：
    1. 字节特征      —— 首部字节序列 + 字节熵
    2. 包长与方向    —— 带符号包长序列及其统计量
    3. 交互行为      —— 到达间隔、突发、上下行比例、握手节奏
    4. 流间关联      —— 同目的主机的流数量、并发度（见 pcap_to_samples）

scapy 缺失或文件非法时退化为确定性伪流，保证界面联调与课堂演示不被环境阻塞。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml.config import BYTE_SEQ_LEN, PKT_SEQ_LEN

try:  # scapy 为可选依赖，缺失时走伪流分支
    from scapy.all import PcapReader  # type: ignore
    from scapy.layers.inet import IP, TCP, UDP  # type: ignore

    SCAPY_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    SCAPY_AVAILABLE = False


STAT_FEATURE_NAMES: List[str] = [
    # 规模类
    "pkt_count", "byte_count", "duration", "pkt_rate", "byte_rate",
    # 包长类
    "len_mean", "len_std", "len_min", "len_max", "len_median",
    "len_p25", "len_p75", "len_skew", "small_pkt_ratio", "large_pkt_ratio",
    # 方向 / 交互行为类
    "fwd_ratio", "bwd_ratio", "fwd_bwd_byte_ratio", "direction_switch_rate",
    "fwd_len_mean", "bwd_len_mean", "burst_count", "mean_burst_size",
    # 时序类
    "iat_mean", "iat_std", "iat_min", "iat_max", "idle_ratio",
    # 字节 / 载荷类
    "payload_entropy", "printable_ratio", "zero_byte_ratio", "high_byte_ratio",
    # TLS 类
    "tls_record_ratio", "handshake_pkt_ratio", "appdata_pkt_ratio",
    "first_appdata_index", "tls13_hint",
    # 流间关联类（由 pcap_to_samples 回填）
    "peer_flow_count", "peer_concurrency", "same_port_flow_ratio",
]

STAT_DIM = len(STAT_FEATURE_NAMES)

FEATURE_GROUPS: Dict[str, Tuple[int, int]] = {
    "规模": (0, 5),
    "包长": (5, 15),
    "方向与交互": (15, 23),
    "时序": (23, 28),
    "字节": (28, 32),
    "TLS": (32, 37),
    "流间关联": (37, STAT_DIM),
}


@dataclass
class Packet:
    ts: float
    length: int
    direction: int  # +1 客户端->服务端，-1 反向
    payload: bytes = b""


@dataclass
class Flow:
    """五元组标识的单向会话（双向合并，以首包方向为正方向）。"""

    src: str
    dst: str
    sport: int
    dport: int
    proto: str
    packets: List[Packet] = field(default_factory=list)

    @property
    def flow_id(self) -> str:
        return f"{self.src}:{self.sport}-{self.dst}:{self.dport}/{self.proto}"


@dataclass
class FlowSample:
    """单条流的模型输入。"""

    flow_id: str
    stats: np.ndarray      # (STAT_DIM,) float32
    pkt_seq: np.ndarray    # (PKT_SEQ_LEN,) float32，带符号归一化包长
    byte_seq: np.ndarray   # (BYTE_SEQ_LEN,) int64，0 为 padding
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# PCAP 解析
# --------------------------------------------------------------------------- #
def extract_flows(pcap_path: Path, max_packets: int = 20000) -> List[Flow]:
    """从 PCAP 中切分双向流；scapy 不可用时返回确定性伪流。"""
    pcap_path = Path(pcap_path)
    if SCAPY_AVAILABLE and pcap_path.exists():
        try:
            return _extract_flows_scapy(pcap_path, max_packets)
        except Exception:
            pass
    return synthesize_flows(pcap_path)


def _extract_flows_scapy(pcap_path: Path, max_packets: int) -> List[Flow]:
    flows: Dict[Tuple, Flow] = {}
    with PcapReader(str(pcap_path)) as reader:
        for index, pkt in enumerate(reader):
            if index >= max_packets:
                break
            if IP not in pkt:
                continue
            ip = pkt[IP]
            if TCP in pkt:
                layer, proto = pkt[TCP], "TCP"
            elif UDP in pkt:
                layer, proto = pkt[UDP], "UDP"
            else:
                continue

            forward_key = (ip.src, ip.dst, int(layer.sport), int(layer.dport), proto)
            reverse_key = (ip.dst, ip.src, int(layer.dport), int(layer.sport), proto)
            if reverse_key in flows:
                flow, direction = flows[reverse_key], -1
            else:
                flow = flows.setdefault(
                    forward_key,
                    Flow(ip.src, ip.dst, int(layer.sport), int(layer.dport), proto),
                )
                direction = 1

            payload = bytes(layer.payload)[:64]
            flow.packets.append(
                Packet(ts=float(pkt.time), length=int(len(pkt)), direction=direction, payload=payload)
            )
    return [flow for flow in flows.values() if len(flow.packets) >= 2]


def synthesize_flows(source: Path, flow_count: int = 4) -> List[Flow]:
    """按文件内容哈希生成确定性伪流。

    仅用于无 scapy / 无真实数据集时的联调，同一文件永远得到同一组流，
    因此界面展示的结果是可复现的。
    """
    source = Path(source)
    seed_material = source.name.encode("utf-8")
    if source.exists():
        with source.open("rb") as handle:
            seed_material += handle.read(65536)
    seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    flows: List[Flow] = []
    for i in range(flow_count):
        packet_count = int(rng.integers(12, 60))
        base_len = float(rng.integers(120, 900))
        timestamp = 0.0
        packets: List[Packet] = []
        for j in range(packet_count):
            timestamp += float(abs(rng.normal(0.02, 0.015))) + 1e-4
            direction = 1 if rng.random() < 0.55 else -1
            length = int(max(60, rng.normal(base_len, base_len * 0.35)))
            payload = bytes(int(x) for x in rng.integers(0, 256, size=48))
            if j < 3:  # 前几包模拟 TLS 握手记录头
                payload = bytes([0x16, 0x03, 0x03]) + payload[3:]
            packets.append(Packet(timestamp, length, direction, payload))
        flows.append(
            Flow(
                src=f"10.0.{seed % 250}.{i + 2}",
                dst=f"93.184.{(seed // 7) % 250}.{i + 10}",
                sport=int(40000 + rng.integers(0, 20000)),
                dport=443,
                proto="TCP",
                packets=packets,
            )
        )
    return flows


# --------------------------------------------------------------------------- #
# 单流特征
# --------------------------------------------------------------------------- #
def flow_to_sample(flow: Flow) -> FlowSample:
    stats = _stat_features(flow)
    return FlowSample(
        flow_id=flow.flow_id,
        stats=stats.astype(np.float32),
        pkt_seq=_packet_length_sequence(flow),
        byte_seq=_byte_sequence(flow),
        meta={
            "src": flow.src,
            "dst": flow.dst,
            "sport": flow.sport,
            "dport": flow.dport,
            "proto": flow.proto,
            "packets": len(flow.packets),
            "bytes": int(sum(p.length for p in flow.packets)),
        },
    )


def _stat_features(flow: Flow) -> np.ndarray:
    packets = flow.packets
    lengths = np.array([p.length for p in packets], dtype=np.float64)
    times = np.array([p.ts for p in packets], dtype=np.float64)
    directions = np.array([p.direction for p in packets], dtype=np.float64)

    duration = float(max(times.max() - times.min(), 1e-6))
    total_bytes = float(lengths.sum())
    fwd_mask, bwd_mask = directions > 0, directions < 0
    fwd_bytes = float(lengths[fwd_mask].sum())
    bwd_bytes = float(lengths[bwd_mask].sum())

    iats = np.diff(times) if len(times) > 1 else np.array([0.0])
    switches = float(np.count_nonzero(np.diff(directions) != 0))
    burst_sizes = _burst_sizes(directions)
    payload = b"".join(p.payload for p in packets)

    std = float(lengths.std())
    skew = float(((lengths - lengths.mean()) ** 3).mean() / (std ** 3)) if std > 1e-6 else 0.0

    values = [
        # 规模
        float(len(packets)), total_bytes, duration,
        len(packets) / duration, total_bytes / duration,
        # 包长
        float(lengths.mean()), std, float(lengths.min()), float(lengths.max()),
        float(np.median(lengths)), float(np.percentile(lengths, 25)),
        float(np.percentile(lengths, 75)), skew,
        float(np.count_nonzero(lengths < 150) / len(lengths)),
        float(np.count_nonzero(lengths > 1200) / len(lengths)),
        # 方向与交互
        float(fwd_mask.sum() / len(packets)), float(bwd_mask.sum() / len(packets)),
        fwd_bytes / (bwd_bytes + 1.0), switches / max(len(packets) - 1, 1),
        float(lengths[fwd_mask].mean()) if fwd_mask.any() else 0.0,
        float(lengths[bwd_mask].mean()) if bwd_mask.any() else 0.0,
        float(len(burst_sizes)), float(np.mean(burst_sizes)) if burst_sizes else 0.0,
        # 时序
        float(iats.mean()), float(iats.std()), float(iats.min()), float(iats.max()),
        float(np.count_nonzero(iats > 1.0) / len(iats)),
        # 字节
        _entropy(payload), _printable_ratio(payload),
        _byte_ratio(payload, lambda b: b == 0), _byte_ratio(payload, lambda b: b > 127),
        # TLS
        *_tls_features(packets),
        # 流间关联占位，由 pcap_to_samples 回填
        0.0, 0.0, 0.0,
    ]
    features = np.asarray(values, dtype=np.float64)
    if features.shape[0] != STAT_DIM:  # 名称表与实现漂移时立刻暴露
        raise ValueError(f"统计特征维度 {features.shape[0]} 与名称表 {STAT_DIM} 不一致")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _burst_sizes(directions: np.ndarray) -> List[int]:
    """连续同方向包视为一个突发。"""
    bursts: List[int] = []
    current = 0
    previous: Optional[float] = None
    for d in directions:
        if previous is None or d == previous:
            current += 1
        else:
            bursts.append(current)
            current = 1
        previous = d
    if current:
        bursts.append(current)
    return bursts


def _tls_features(packets: Sequence[Packet]) -> List[float]:
    """TLS 记录层粗粒度特征：握手 / 应用数据占比与首个 AppData 位置。"""
    total = len(packets)
    record, handshake, appdata = 0, 0, 0
    first_appdata = -1
    tls13_hint = 0.0
    for index, pkt in enumerate(packets):
        head = pkt.payload[:3]
        if len(head) < 3 or head[0] not in (0x14, 0x15, 0x16, 0x17):
            continue
        record += 1
        if head[1] != 0x03:
            continue
        if head[0] == 0x16:
            handshake += 1
            # TLS 1.3 记录层版本固定伪装为 0x0303，握手中通过扩展协商
            if head[2] == 0x03:
                tls13_hint = 1.0
        elif head[0] == 0x17:
            appdata += 1
            if first_appdata < 0:
                first_appdata = index
    return [
        record / total,
        handshake / total,
        appdata / total,
        float(first_appdata) / total if first_appdata >= 0 else -1.0,
        tls13_hint,
    ]


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-(probs * np.log2(probs)).sum())


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    return sum(1 for b in data if 32 <= b < 127) / len(data)


def _byte_ratio(data: bytes, predicate) -> float:
    if not data:
        return 0.0
    return sum(1 for b in data if predicate(b)) / len(data)


def _packet_length_sequence(flow: Flow, length: int = PKT_SEQ_LEN) -> np.ndarray:
    """带方向符号的包长序列，除以 1500 归一化后补零。"""
    seq = [p.direction * min(p.length, 1500) / 1500.0 for p in flow.packets[:length]]
    seq.extend([0.0] * (length - len(seq)))
    return np.asarray(seq, dtype=np.float32)


def _byte_sequence(flow: Flow, length: int = BYTE_SEQ_LEN) -> np.ndarray:
    """前若干包载荷拼接的字节序列，取值 1-256，0 保留给 padding。"""
    raw = b"".join(p.payload for p in flow.packets[:8])[:length]
    seq = [b + 1 for b in raw]
    seq.extend([0] * (length - len(seq)))
    return np.asarray(seq, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 流间关联
# --------------------------------------------------------------------------- #
def pcap_to_samples(pcap_path: Path, max_flows: int = 64) -> List[FlowSample]:
    """PCAP -> 样本列表，并回填流间关联特征。"""
    flows = extract_flows(pcap_path)[:max_flows]
    samples = [flow_to_sample(flow) for flow in flows]
    _augment_cross_flow(flows, samples)
    return samples


def _augment_cross_flow(flows: Sequence[Flow], samples: Sequence[FlowSample]) -> None:
    if not flows:
        return
    start = FEATURE_GROUPS["流间关联"][0]
    peer_counter: Dict[str, int] = {}
    port_counter: Dict[int, int] = {}
    for flow in flows:
        peer_counter[flow.dst] = peer_counter.get(flow.dst, 0) + 1
        port_counter[flow.dport] = port_counter.get(flow.dport, 0) + 1

    windows = [(f.packets[0].ts, f.packets[-1].ts) for f in flows]
    for flow, sample, (begin, end) in zip(flows, samples, windows):
        concurrency = sum(1 for b, e in windows if b <= end and e >= begin)
        sample.stats[start] = float(peer_counter[flow.dst])
        sample.stats[start + 1] = float(concurrency)
        sample.stats[start + 2] = port_counter[flow.dport] / len(flows)
