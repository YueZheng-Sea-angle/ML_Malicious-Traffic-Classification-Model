"""算法层公共配置：类别定义、序列长度、路径常量。

产品分类口径采用 DataCon2021-ETA part1 的 T1（11 类加密代理/隧道工具形态识别），
类别顺序与官方标签 id 0-10 完全一致，训练与推理必须共享本表以免标签错位；
代理使用检测（T2，tunnel/normal）等二元口径保留在 ml/research 层。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
UPLOAD_DIR = ARTIFACT_DIR / "uploads"

# 分类类别，顺序即标签 id，训练与推理必须一致（DataCon part1 官方 11 类）
CLASS_NAMES = [
    "openvpn-udp",    # OpenVPN UDP 隧道
    "psiphon-tls",    # Psiphon TLS
    "v2ray",          # V2Ray
    "clash",          # Clash
    "lantern",        # Lantern
    "openvpn-tls",    # OpenVPN TLS 隧道
    "firefox",        # Firefox 直连（正常对照）
    "psiphon-tcp",    # Psiphon TCP
    "wireguard-udp",  # WireGuard UDP 隧道
    "shadowsocks",    # Shadowsocks
    "netch",          # Netch
]

CLASS_NAMES_ZH = {
    "openvpn-udp": "OpenVPN（UDP）",
    "psiphon-tls": "Psiphon（TLS）",
    "v2ray": "V2Ray",
    "clash": "Clash",
    "lantern": "Lantern",
    "openvpn-tls": "OpenVPN（TLS）",
    "firefox": "Firefox 直连",
    "psiphon-tcp": "Psiphon（TCP）",
    "wireguard-udp": "WireGuard（UDP）",
    "shadowsocks": "Shadowsocks",
    "netch": "Netch",
}

NUM_CLASSES = len(CLASS_NAMES)

# 三路输入的定长规格
PKT_SEQ_LEN = 32     # 包长方向序列长度
BYTE_SEQ_LEN = 256   # 首部字节序列长度
BYTE_VOCAB = 257     # 0-255 字节 + 1 个 padding 位

DEFAULT_CHECKPOINT = MODEL_DIR / "malflow_datacon_tools.pt"
