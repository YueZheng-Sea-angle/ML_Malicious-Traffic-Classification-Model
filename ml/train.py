"""离线训练入口。

用法：
    # 合成数据（数据集到位前打通链路）
    python -m ml.train --synthetic --epochs 8

    # 真实数据集
    python -m ml.train --data-dir data/raw --epochs 30 --batch-size 64

产物：
    artifacts/models/malflow_cnn_bilstm.pt   模型权重 + 标准化参数 + 特征掩码
    artifacts/models/feature_report.json     特征贡献评估报告
    artifacts/models/train_metrics.json      训练与验证指标
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import CLASS_NAMES, DEFAULT_CHECKPOINT, MODEL_DIR, NUM_CLASSES
from ml.data.dataset import (
    build_dataset_from_pcaps,
    make_synthetic_dataset,
    split_dataset,
)
from ml.features.selection import FeatureSelector
from ml.models.cnn_bilstm import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="恶意流量分类模型训练")
    parser.add_argument("--data-dir", type=Path, default=None, help="按类别分目录存放的 PCAP 根目录")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据集")
    parser.add_argument("--samples-per-class", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--coverage", type=float, default=0.95, help="特征累计贡献覆盖率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    batch = (
        build_dataset_from_pcaps(args.data_dir)
        if args.data_dir and not args.synthetic
        else make_synthetic_dataset(args.samples_per_class, seed=args.seed)
    )
    train_set, val_set = split_dataset(batch, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[数据] 训练 {len(train_set)} 条 / 验证 {len(val_set)} 条 / 类别 {NUM_CLASSES}")

    # 统计特征标准化参数只能来自训练集，避免验证集信息泄漏
    mean = train_set.stats.mean(axis=0)
    std = train_set.stats.std(axis=0) + 1e-6

    selector = FeatureSelector(coverage=args.coverage).fit(
        (train_set.stats - mean) / std, train_set.labels
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    selector.save(MODEL_DIR / "feature_report.json")
    print(f"[特征] 自适应保留 {len(selector.selected_)}/{train_set.stats.shape[1]} 维，Top5："
          f"{[s.name for s in selector.top(5)]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES).to(device)
    model.set_prior_mask(torch.from_numpy(selector.mask()))

    train_loader = DataLoader(
        _to_tensor_dataset(train_set, mean, std), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(_to_tensor_dataset(val_set, mean, std), batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history, best_acc = [], 0.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"[训练] epoch {epoch:02d}/{args.epochs} loss={train_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"({time.time() - started:.1f}s)"
        )
        if metrics["accuracy"] >= best_acc:
            best_acc = metrics["accuracy"]
            _save_checkpoint(args.output, model, mean, std, selector, metrics)

    (MODEL_DIR / "train_metrics.json").write_text(
        json.dumps({"best_accuracy": best_acc, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[完成] 最佳验证准确率 {best_acc:.4f}，权重已保存至 {args.output}")


# --------------------------------------------------------------------------- #
def _to_tensor_dataset(batch, mean: np.ndarray, std: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(((batch.stats - mean) / std).astype(np.float32)),
        torch.from_numpy(batch.pkt.astype(np.float32)),
        torch.from_numpy(batch.byte.astype(np.int64)),
        torch.from_numpy(batch.labels.astype(np.int64)),
    )


def _train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for stats, pkt, byte, labels in loader:
        stats, pkt, byte, labels = (t.to(device) for t in (stats, pkt, byte, labels))
        optimizer.zero_grad()
        loss = criterion(model(stats, pkt, byte)["logits"], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * labels.size(0)
        count += labels.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, object]:
    model.eval()
    preds, targets = [], []
    for stats, pkt, byte, labels in loader:
        stats, pkt, byte = (t.to(device) for t in (stats, pkt, byte))
        preds.append(model(stats, pkt, byte)["logits"].argmax(dim=-1).cpu().numpy())
        targets.append(labels.numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(targets) if targets else np.array([])
    return {
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "macro_f1": _macro_f1(y_true, y_pred),
        "per_class_recall": _per_class_recall(y_true, y_pred),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in range(NUM_CLASSES):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    result = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = y_true == c
        result[name] = float((y_pred[mask] == c).mean()) if mask.any() else 0.0
    return result


def _save_checkpoint(path: Path, model, mean, std, selector: FeatureSelector, metrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "stat_mean": mean,
            "stat_std": std,
            "feature_mask": selector.mask(),
            "selected_features": [selector.feature_names[i] for i in selector.selected_],
            "metrics": metrics,
            "version": "0.1.0",
        },
        path,
    )


if __name__ == "__main__":
    main()
