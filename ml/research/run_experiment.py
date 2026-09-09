"""DataCon 研究实验训练器：文件级划分 + 三路特征 + MalFlowNet 端到端。

任务口径（对应《特征方案与详细设计.md》§4.2 任务矩阵）：
    --task tools   T1：11 类工具形态识别（官方标签）
    --task agent   T2：场景假设二元（firefox=normal，6 类隧道=tunnel，灰色剔除）

用法示例：
    python -m ml.research.run_experiment --task tools --real-per-class 0 --epochs 3
    python -m ml.research.run_experiment --task agent --real-per-class 2 --epochs 5

训练/评估协议：文件级分层划分（ml/research/split.py，防同源泄漏）、统计特征
标准化仅取自训练集、FeatureSelector 掩码注入门控、checkpoint 写入 task_scope。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import MODEL_DIR
from ml.data.dataset import save_dataset
from ml.features.selection import FeatureSelector
from ml.models.cnn_bilstm import build_model
from ml.research.datacon import DEFAULT_RESEARCH_DIR, DataConLoader, agent_binary
from ml.research.split import split_by_file
from ml.train import _save_checkpoint, _to_tensor_dataset, _train_one_epoch, evaluate

TASK_SCOPE = {"tools": "tools11", "agent": "agent2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataCon 研究实验（MalFlowNet 端到端）")
    p.add_argument("--task", choices=list(TASK_SCOPE), default="tools")
    p.add_argument("--real-per-class", type=int, default=0,
                   help="每类补充 real_data 文件数(0=只用 sample)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--class-weights", action="store_true", help="按类别频数倒数加权损失")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", type=Path, default=None, help="特征缓存 npz 路径")
    p.add_argument("--rebuild", action="store_true", help="忽略缓存重建特征")
    p.add_argument("--out", type=Path, default=None, help="checkpoint 输出路径")
    return p.parse_args()


def _class_weights_tensor(labels: np.ndarray, n_classes: int,
                          device) -> Optional[torch.Tensor]:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weight = counts.sum() / (n_classes * counts)
    weight = weight / weight.sum() * n_classes
    return torch.from_numpy(weight.astype(np.float32)).to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scope = TASK_SCOPE[args.task]
    cache = args.cache or Path(DEFAULT_RESEARCH_DIR) / f"part1_sample_rpc{args.real_per_class}.npz"

    if args.rebuild or not cache.exists():
        batch = DataConLoader().build(include_sample=True, real_per_class=args.real_per_class,
                                      cache=None, seed=args.seed)
        save_dataset(batch, cache)
    else:
        from ml.data.dataset import load_dataset
        batch = load_dataset(cache)

    if args.task == "agent":
        batch = agent_binary(batch)
        scope = "agent2"
    class_names = list(batch.class_names)
    print(f"[实验] task={args.task} 类别={class_names} 样本={len(batch)}")

    train_set, val_set = split_by_file(batch, val_ratio=args.val_ratio, seed=args.seed)

    mean = train_set.stats.mean(axis=0)
    std = train_set.stats.std(axis=0) + 1e-6
    selector = FeatureSelector(coverage=args.coverage).fit(
        (train_set.stats - mean) / std, train_set.labels)
    print(f"[特征] 自适应保留 {len(selector.selected_)}/40 维，Top5："
          f"{[s.name for s in selector.top(5)]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names)).to(device)
    model.set_prior_mask(torch.from_numpy(selector.mask()))

    train_loader = DataLoader(_to_tensor_dataset(train_set, mean, std),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(_to_tensor_dataset(val_set, mean, std),
                            batch_size=args.batch_size)

    weight = _class_weights_tensor(train_set.labels, len(class_names), device) \
        if args.class_weights else None
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history, best_acc = [], 0.0
    for epoch in range(1, args.epochs + 1):
        loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device, class_names=class_names)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(loss, 4), **metrics})
        print(f"[训练] epoch {epoch:02d}/{args.epochs} loss={loss:.4f} "
              f"val_acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
        if metrics["accuracy"] >= best_acc:
            best_acc = metrics["accuracy"]
            checkpoint = args.out or Path(MODEL_DIR) / f"malflow_datacon_{args.task}.pt"
            _save_checkpoint(checkpoint, model, mean, std, selector, metrics,
                             class_names=class_names, task_scope=scope)
            print(f"  ↳ 已保存最佳权重 {checkpoint}（best_acc={best_acc:.4f}）")

    report = {"task": scope, "class_names": class_names, "best_accuracy": best_acc,
              "selected_features": [selector.feature_names[i] for i in selector.selected_],
              "history": history,
              "per_class_recall": history[-1]["per_class_recall"]}
    out = Path(DEFAULT_RESEARCH_DIR) / f"experiment_{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[实验] 报告已写入 {out}；逐类召回：")
    for name, r in report["per_class_recall"].items():
        print(f"  {name:<16} recall={r:.4f}")


if __name__ == "__main__":
    main()
