# 算法层说明

## 数据流

```
PCAP ──extractor.extract_flows──> Flow(五元组 + 包序列)
        │
        ├── _stat_features        40 维统计特征（规模/包长/方向交互/时序/字节/TLS/流间关联）
        ├── _packet_length_sequence  32 维带符号包长序列
        └── _byte_sequence           256 维首部字节序列
                    │
                    ├── FeatureSelector  离线评估贡献 -> 0/1 掩码
                    └── MalFlowNet       三路编码 + 门控融合 -> 6 类概率
```

## 类别定义

类别顺序即标签 id，训练与推理共用 `ml/config.py` 的 `CLASS_NAMES`：

| id | label | 中文 |
|----|-------|------|
| 0 | benign | 正常流量 |
| 1 | botnet | 僵尸网络 |
| 2 | ransomware | 勒索软件 |
| 3 | trojan | 木马 |
| 4 | cryptomining | 挖矿 |
| 5 | ddos | DDoS 攻击 |

正式数据集的标签体系确认后，只需修改 `CLASS_NAMES` 与 `CLASS_NAMES_ZH`，
其余代码无需改动。

## 常用命令

```bash
# 项目根目录（project/）下执行
python -m ml.train --synthetic --epochs 8          # 合成数据训练
python -m ml.train --data-dir data/raw --epochs 30 # 真实数据训练
python -m ml.predict --file sample.pcap            # 单文件推理
python -m ml.models.cnn_bilstm                     # 网络结构形状自检
```

## 训练产物

| 文件 | 内容 |
|------|------|
| `artifacts/models/malflow_cnn_bilstm.pt` | 权重、标准化参数、特征掩码、指标 |
| `artifacts/models/feature_report.json` | 各特征贡献度与入选列表 |
| `artifacts/models/train_metrics.json` | 逐轮训练/验证指标 |

## 待办（第 2 周）

- [ ] 接入正式数据集，替换合成样本
- [ ] 补充混淆矩阵与 PR 曲线绘制脚本
- [ ] 特征选择结果与门控权重的一致性分析（写入概要设计）
- [ ] 模型轻量化以满足在线推理时延要求
