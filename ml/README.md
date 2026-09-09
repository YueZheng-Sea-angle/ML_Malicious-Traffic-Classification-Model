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
                    └── LightGBM A/B      统计(±n-gram) -> 11 类代理/隧道工具概率
```

## 类别定义

类别顺序即标签 id，与 DataCon2021-ETA part1 官方 11 类顺序一致，训练与推理共用
`ml/config.py` 的 `CLASS_NAMES`：

| id | label | 中文 |
|----|-------|------|
| 0 | openvpn-udp | OpenVPN（UDP） |
| 1 | psiphon-tls | Psiphon（TLS） |
| 2 | v2ray | V2Ray |
| 3 | clash | Clash |
| 4 | lantern | Lantern |
| 5 | openvpn-tls | OpenVPN（TLS） |
| 6 | firefox | Firefox 直连 |
| 7 | psiphon-tcp | Psiphon（TCP） |
| 8 | wireguard-udp | WireGuard（UDP） |
| 9 | shadowsocks | Shadowsocks |
| 10 | netch | Netch |

需要调整任务口径时，只需修改 `CLASS_NAMES` 与 `CLASS_NAMES_ZH`，其余代码无需改动。

## 常用命令

```bash
# 项目根目录（project/）下执行
python -m ml.train --data-dir data/raw --epochs 30 # 按类别目录组织的真实数据训练
python -m ml.predict --file sample.pcap            # 单文件推理
python -m ml.models.cnn_bilstm                     # 网络结构形状自检

# DataCon T1 A/B 双模型（同一训练器，产出基线与 n-gram 增强两组权重）
python -m ml.research.run_experiment --real-per-class 5 --epochs 5
```

## 训练产物

| 文件 | 内容 |
|------|------|
| `artifacts/models/malflow_datacon_tools.pt` | T1 A 组基线权重、标准化参数、特征掩码、指标 |
| `artifacts/models/malflow_datacon_tools_ngram.pt` | T1 B 组 n-gram 增强权重（含 n-gram 词表） |
| `artifacts/research/ab_tools_report.json` | A/B 双模型验证集对比报告 |
| `artifacts/models/feature_report.json` | 各特征贡献度与入选列表 |

## 待办

- [ ] 全量 real_data（1000 文件）扩充训练样本并复核 T1 指标
- [ ] 补充混淆矩阵与 PR 曲线绘制脚本
- [ ] 特征选择结果与门控权重的一致性分析（写入概要设计）
- [ ] 模型轻量化以满足在线推理时延要求
