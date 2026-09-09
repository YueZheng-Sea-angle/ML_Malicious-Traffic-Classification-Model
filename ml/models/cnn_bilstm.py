"""MalFlowNet：三路输入融合的加密代理/隧道流量工具分类网络。

    字节序列 --> Embedding + 1D-CNN      局部字节模式（加密载荷指纹）
    包长序列 --> BiLSTM + 池化           包长/方向的时序交互行为
    统计特征 --> 门控 MLP                多维统计量，门控实现自适应特征加权

三路特征拼接后送入分类头。门控向量 gate 可导出用于「特征贡献」可视化，
与 FeatureSelector 的离线评估互相印证。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ml.config import BYTE_VOCAB, NUM_CLASSES, PKT_SEQ_LEN
from ml.features.extractor import STAT_DIM


class ByteCNN(nn.Module):
    def __init__(self, embed_dim: int = 32, out_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(BYTE_VOCAB, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        x = self.embedding(byte_seq).transpose(1, 2)
        return self.conv(x).squeeze(-1)


class PacketBiLSTM(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        # 输出维度 = 2(双向) * hidden * 2(mean+max 池化) = 128
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, pkt_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(pkt_seq.unsqueeze(-1))
        return torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)


class GatedStatEncoder(nn.Module):
    """统计特征编码器，内置可学习门控实现自适应特征筛选。"""

    def __init__(self, in_dim: int = STAT_DIM, out_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 96), nn.ReLU(), nn.Dropout(0.2), nn.Linear(96, out_dim), nn.ReLU()
        )

    def forward(self, stats: torch.Tensor, prior_mask: Optional[torch.Tensor] = None):
        x = self.norm(stats)
        gate = self.gate(x)
        if prior_mask is not None:
            # 离线选择结果作为先验：未入选特征保留 30% 通路，避免过早剪枝
            gate = gate * (0.3 + 0.7 * prior_mask)
        return self.mlp(x * gate), gate


class NGramEncoder(nn.Module):
    """载荷字节 n-gram 编码器（B 组增强特征路）。

    输入为逐流的 n-gram 特征向量（TF-IDF 计数，dim=K，由离线 vectorizer 生成），
    经 MLP 投影为定长 64 维；仅在 checkpoint 启用 n-gram（ngram_dim>0）时存在。
    """

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, out_dim), nn.ReLU()
        )

    def forward(self, ngram: torch.Tensor) -> torch.Tensor:
        return self.mlp(ngram)


class MalFlowNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 ngram_dim: int = 0) -> None:
        super().__init__()
        self.ngram_dim = int(ngram_dim)
        self.byte_encoder = ByteCNN()
        self.pkt_encoder = PacketBiLSTM()
        self.stat_encoder = GatedStatEncoder()
        self.ngram_encoder = NGramEncoder(self.ngram_dim) if self.ngram_dim > 0 else None
        fusion_dim = 128 + 128 + 64 + (64 if self.ngram_encoder is not None else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.register_buffer("prior_mask", torch.ones(STAT_DIM))

    def set_prior_mask(self, mask: torch.Tensor) -> None:
        """写入 FeatureSelector 产出的 0/1 掩码。"""
        self.prior_mask.copy_(mask.to(self.prior_mask.device).float())

    def forward(
        self, stats: torch.Tensor, pkt_seq: torch.Tensor, byte_seq: torch.Tensor,
        ngram: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        stat_vec, gate = self.stat_encoder(stats, self.prior_mask.expand(stats.size(0), -1))
        parts = [self.byte_encoder(byte_seq), self.pkt_encoder(pkt_seq), stat_vec]
        if self.ngram_encoder is not None:
            if ngram is None:
                raise ValueError("模型启用了 n-gram 输入，forward 缺少 ngram 张量")
            parts.append(self.ngram_encoder(ngram))
        fused = torch.cat(parts, dim=-1)
        return {"logits": self.classifier(fused), "gate": gate, "embedding": fused}


def build_model(num_classes: int = NUM_CLASSES, ngram_dim: int = 0) -> MalFlowNet:
    return MalFlowNet(num_classes=num_classes, ngram_dim=ngram_dim)


if __name__ == "__main__":  # 形状自检
    model = build_model()
    batch = 4
    kwargs = {
        "stats": torch.randn(batch, STAT_DIM),
        "pkt_seq": torch.randn(batch, PKT_SEQ_LEN),
        "byte_seq": torch.randint(0, BYTE_VOCAB, (batch, 256)),
    }
    print("baseline:", {k: tuple(v.shape) for k, v in model(**kwargs).items()})
    model_b = build_model(ngram_dim=1000)
    kwargs["ngram"] = torch.randn(batch, 1000)
    print("ngram  :", {k: tuple(v.shape) for k, v in model_b(**kwargs).items()})
