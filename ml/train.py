"""离线训练入口。

用法：
    # 按类别目录组织 PCAP（类别名 = ml.config.CLASS_NAMES）
    python -m ml.train --data-dir data/raw --epochs 30 --batch-size 64

    # DataCon T1（11 类代理工具）/ T2（tunnel/normal）请走 research 装载器：
    python -m ml.research.run_experiment --task tools
    python -m ml.research.run_experiment --task agent

产物：
    artifacts/models/malflow_datacon_tools.pt  模型权重 + 标准化参数 + 特征掩码
    artifacts/models/feature_report.json       特征贡献评估报告
    artifacts/models/train_metrics.json        训练与验证指标
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

from ml.config import CLASS_NAMES, DEFAULT_CHECKPOINT, MODEL_DIR
from ml.data.dataset import build_dataset_from_pcaps, split_dataset
from ml.features.selection import FeatureSelector
from ml.models.cnn_bilstm import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加密代理/隧道工具分类模型训练（T1）")
    parser.add_argument("--data-dir", type=Path, default=None, help="按类别分目录存放的 PCAP 根目录")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--coverage", type=float, default=0.95, help="特征累计贡献覆盖率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-scope", type=str, default="tools11",
                        help="任务口径标识，写入 checkpoint：tools11/agent2")
    parser.add_argument("--class-weights", action="store_true",
                        help="按训练集类别频数倒数加权交叉熵（类别不平衡时启用）")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir:
        raise SystemExit("请通过 --data-dir 指定按类别组织的 PCAP 根目录")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    batch = build_dataset_from_pcaps(args.data_dir)
    class_names = list(batch.class_names)
    train_set, val_set = split_dataset(batch, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[数据] 训练 {len(train_set)} 条 / 验证 {len(val_set)} 条 / 类别 {len(class_names)}")

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
    model = build_model(len(class_names)).to(device)
    model.set_prior_mask(torch.from_numpy(selector.mask()))

    train_loader = DataLoader(
        _to_tensor_dataset(train_set, mean, std), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(_to_tensor_dataset(val_set, mean, std), batch_size=args.batch_size)

    if args.class_weights:
        counts = np.bincount(train_set.labels, minlength=len(class_names)).astype(np.float64)
        counts[counts == 0] = 1.0  # 训练集缺失的类别不参与加权
        weight = counts.sum() / (len(class_names) * counts)
        weight = weight / weight.sum() * len(class_names)  # 保持与原损失量级一致
        print("[权重] 类别加权："
              + ", ".join(f"{n}={w:.2f}" for n, w in zip(class_names, weight)))
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05,
        weight=torch.from_numpy(weight.astype(np.float32)).to(device) if weight is not None else None,
    )
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
            _save_checkpoint(args.output, model, mean, std, selector, metrics,
                             class_names=class_names, task_scope=args.task_scope)

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
def evaluate(model, loader, device, class_names: Optional[List[str]] = None) -> Dict[str, object]:
    """评估模型：accuracy / macro-F1 / 逐类召回。class_names 为空时按数据推断类别数。"""
    model.eval()
    preds, targets = [], []
    for stats, pkt, byte, labels in loader:
        stats, pkt, byte = (t.to(device) for t in (stats, pkt, byte))
        preds.append(model(stats, pkt, byte)["logits"].argmax(dim=-1).cpu().numpy())
        targets.append(labels.numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(targets) if targets else np.array([])
    names = list(class_names) if class_names else _infer_class_names(y_true, y_pred)
    return {
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "macro_f1": _macro_f1(y_true, y_pred),
        "per_class_recall": _per_class_recall(y_true, y_pred, names),
    }


def _infer_class_names(y_true: np.ndarray, y_pred: np.ndarray) -> List[str]:
    """按数据集中出现的类别 id 推断类别名（仅为缺失 class_names 时的兜底）。"""
    if not len(y_true):
        return list(CLASS_NAMES)
    n = int(max(y_true.max(), y_pred.max())) + 1 if len(y_pred) else int(y_true.max()) + 1
    return [str(CLASS_NAMES[i]) if i < len(CLASS_NAMES) else f"class_{i}" for i in range(n)]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not len(y_true):
        return 0.0
    n_classes = int(max(y_true.max(), y_pred.max())) + 1 if len(y_pred) else int(y_true.max()) + 1
    scores = []
    for c in range(n_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: Optional[List[str]] = None) -> Dict[str, float]:
    names = list(class_names) if class_names else list(CLASS_NAMES)
    result = {}
    for c, name in enumerate(names):
        mask = y_true == c
        result[name] = float((y_pred[mask] == c).mean()) if mask.any() else 0.0
    return result


def _save_checkpoint(path: Path, model, mean, std, selector: FeatureSelector, metrics,
                     class_names: Optional[List[str]] = None,
                     task_scope: str = "tools11") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(class_names) if class_names else CLASS_NAMES,
            "task_scope": task_scope,
            "stat_mean": mean,
            "stat_std": std,
            "feature_mask": selector.mask(),
            "selected_features": [selector.feature_names[i] for i in selector.selected_],
            "metrics": metrics,
            "version": "0.2.0",
        },
        path,
    )


if __name__ == "__main__":
    main()
