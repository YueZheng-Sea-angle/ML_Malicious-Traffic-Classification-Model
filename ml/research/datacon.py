"""DataCon2021-ETA(加密代理流量)研究数据集装载与特征提取。

只负责「解析 + 标签映射 + 特征缓存」，产品特征管线仍复用 ml/features/extractor。
标签依据 docs/数据集分析.md：part1 共 11 类工具(0-10)，part1_label.txt 记录
real_data 标签，sample 目录文件名 label_n.pcap 自带标签。类别语义为软件身份，
不含恶意/正常定性；二元口径由 agent_binary() 按场景假设给出。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml.config import PROJECT_ROOT
from ml.data.dataset import SampleBatch, load_dataset, save_dataset
from ml.features.extractor import pcap_to_samples

# DataCon part1 官方 11 类（数组下标即官方标签号 0-10）
TOOLS_11 = [
    "openvpn-udp", "psiphon-tls", "v2ray", "clash", "lantern",
    "openvpn-tls", "firefox", "psiphon-tcp", "wireguard-udp", "shadowsocks", "netch",
]
TOOLS_11_ZH = [
    "OpenVPN(UDP)", "Psiphon(TLS)", "V2Ray", "Clash", "Lantern",
    "OpenVPN(TLS)", "Firefox 直连", "Psiphon(TCP)", "WireGuard(UDP)", "Shadowsocks", "Netch",
]

# 场景假设(T2)：疑似代理隧道 / 正常直连 / 灰色(合规常用，训练时排除)
TUNNEL_POS = {"psiphon-tls", "v2ray", "clash", "lantern", "psiphon-tcp", "shadowsocks", "netch"}
NORMAL_NEG = {"firefox"}
GRAY_CLS = {"openvpn-udp", "openvpn-tls", "wireguard-udp"}

DEFAULT_DATA_ROOT = PROJECT_ROOT / "DataCon2021加密代理流量数据集" / "datacon2021_eta"
DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "artifacts" / "research"


class DataConLoader:
    """DataCon part1 装载器：sample(样例) + real_data(带标签目标流量)。"""

    def __init__(self, root: Path = DEFAULT_DATA_ROOT) -> None:
        self.root = Path(root)
        self.part1 = self.root / "part1"
        self._real_label: Dict[str, int] = {}
        self._real_loaded = False

    def _label_path(self) -> Path:
        # 官方两种布局兼容：datacon2021_eta/part1/part1_label.txt 或 part1 目录上一级
        candidates = [
            self.part1 / "part1_label.txt",
            self.root / "part1_label.txt",
            self.root / "part1" / "label.txt",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"未找到 part1 标签文件，已查找：{', '.join(str(p) for p in candidates)}"
        )

    # ------------------------------------------------------------------ #
    def _load_real_label(self) -> Dict[str, int]:
        if not self._real_loaded:
            for line in self._label_path().read_text(encoding="utf-8").splitlines():
                fn, c = line.split()
                self._real_label[fn] = int(c)
            self._real_loaded = True
        return self._real_label

    def sample_items(self) -> List[Tuple[Path, int]]:
        """sample 目录文件（文件名 label_n.pcap）。"""
        items = []
        for p in sorted((self.part1 / "sample").glob("*.pcap")):
            m = re.match(r"^(\d+)_", p.stem)
            if m and int(m.group(1)) < len(TOOLS_11):
                items.append((p, int(m.group(1))))
        return items

    def real_items(self, per_class: Optional[int] = None, seed: int = 0) -> List[Tuple[Path, int]]:
        """real_data 文件（按 per-class 随机抽样；None 表示全部）。"""
        label = self._load_real_label()
        grouped: Dict[int, List[Path]] = {}
        for p in sorted((self.part1 / "real_data").glob("*.pcap")):
            c = label.get(p.name)
            if c is not None:
                grouped.setdefault(c, []).append(p)
        items: List[Tuple[Path, int]] = []
        for c in sorted(grouped):
            paths = grouped[c]
            if per_class is not None:
                rng = np.random.default_rng(seed)
                paths = list(rng.choice(paths, size=min(per_class, len(paths)), replace=False))
            items.extend((p, c) for p in paths)
        return items

    # ------------------------------------------------------------------ #
    def build(self, include_sample: bool = True, real_per_class: Optional[int] = None,
              max_flows_per_file: int = 64, cache: Optional[Path] = None,
              seed: int = 0) -> SampleBatch:
        """全量特征提取为 SampleBatch（flow_ids 以「文件stem::flow_id」命名以便文件级分组）。

        特征缓存命中时直接读取 npz，避免重复解析。
        """
        if cache is not None and Path(cache).exists():
            print(f"[datacon] 命中特征缓存 {cache}")
            return load_dataset(cache)

        files: List[Tuple[Path, int]] = []
        if include_sample:
            files.extend(self.sample_items())
        if real_per_class:
            files.extend(self.real_items(per_class=real_per_class, seed=seed))

        stats, pkts, bytes_, labels, flow_ids = [], [], [], [], []
        fail, n_file = 0, 0
        for p, c in files:
            try:
                samples = pcap_to_samples(p, max_flows=max_flows_per_file)
            except Exception:  # 个别坏文件不影响整体构建
                fail += 1
                continue
            if not samples:
                continue
            n_file += 1
            for s in samples:
                stats.append(s.stats)
                pkts.append(s.pkt_seq)
                bytes_.append(s.byte_seq)
                labels.append(c)
                flow_ids.append(f"{p.stem}::{s.flow_id}")

        if not labels:
            raise FileNotFoundError(f"{self.root} 下未解析到任何可用样本")
        batch = _stack_batch(stats, pkts, bytes_, labels, flow_ids, TOOLS_11)
        print(f"[datacon] 文件 {n_file}（跳过 {fail}）→ 样本 {len(batch)}，"
              f"每类文件数：{_per_class_file_count(files)}")
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            save_dataset(batch, cache)
        return batch


def _stack_batch(stats, pkts, bytes_, labels, flow_ids, class_names: Sequence[str]) -> SampleBatch:
    from ml.data.dataset import _stack

    return _stack(stats, pkts, bytes_, labels, flow_ids, class_names=list(class_names))


def _per_class_file_count(files: List[Tuple[Path, int]]) -> str:
    counts: Dict[int, int] = {}
    for _, c in files:
        counts[c] = counts.get(c, 0) + 1
    return "{" + ", ".join(f"{i}:{counts.get(i, 0)}" for i in sorted(counts)) + "}"


def flow_file_key(flow_id: str) -> str:
    """从 flow_id 还原文件分组键（flow_id 格式：<文件stem>::<流id>）。"""
    return flow_id.split("::", 1)[0]


def agent_binary(batch: SampleBatch) -> SampleBatch:
    """场景假设 T2：firefox→normal(0)，6 类隧道工具→tunnel(1)；灰色类(openvpn/wireguard)剔除。"""
    keep_idx, new_labels = [], []
    for i, lab in enumerate(batch.labels):
        name = batch.class_names[int(lab)]
        if name in NORMAL_NEG:
            keep_idx.append(i)
            new_labels.append(0)
        elif name in TUNNEL_POS:
            keep_idx.append(i)
            new_labels.append(1)
        # 灰色类(openvpn-*/wireguard-*)直接丢弃
    if not keep_idx:
        raise ValueError("二元口径下无保留样本，请检查类别集合")
    idx = np.asarray(keep_idx, dtype=np.int64)
    return SampleBatch(
        stats=batch.stats[idx],
        pkt=batch.pkt[idx],
        byte=batch.byte[idx],
        labels=np.asarray(new_labels, dtype=np.int64),
        flow_ids=[batch.flow_ids[i] for i in idx],
        class_names=["normal", "tunnel"],
    )
