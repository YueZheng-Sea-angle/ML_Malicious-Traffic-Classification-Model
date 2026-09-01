"""算法层公共配置：类别定义、序列长度、路径常量。

后端与训练脚本共享本文件，避免类别顺序不一致导致标签错位。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
UPLOAD_DIR = ARTIFACT_DIR / "uploads"

# 分类类别，顺序即标签 id，训练与推理必须一致
CLASS_NAMES = [
    "benign",       # 正常加密流量
    "botnet",       # 僵尸网络 C&C
    "ransomware",   # 勒索软件回传
    "trojan",       # 木马下载/控制
    "cryptomining",  # 挖矿
    "ddos",         # 拒绝服务攻击
]

CLASS_NAMES_ZH = {
    "benign": "正常流量",
    "botnet": "僵尸网络",
    "ransomware": "勒索软件",
    "trojan": "木马",
    "cryptomining": "挖矿",
    "ddos": "DDoS 攻击",
}

NUM_CLASSES = len(CLASS_NAMES)

# 三路输入的定长规格
PKT_SEQ_LEN = 32     # 包长方向序列长度
BYTE_SEQ_LEN = 256   # 首部字节序列长度
BYTE_VOCAB = 257     # 0-255 字节 + 1 个 padding 位

DEFAULT_CHECKPOINT = MODEL_DIR / "malflow_cnn_bilstm.pt"
