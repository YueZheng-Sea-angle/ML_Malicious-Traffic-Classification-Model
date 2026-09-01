# 恶意流量分类系统（MalFlow）

> 基于深度学习方法的 TLS 1.3 加密恶意流量分类模型

当前为 **v0.1.0 原型**：三层链路（前端 → 后端 → 算法）已打通，模型权重缺失时
自动退化为演示推理，便于在数据集到位前完成界面与流程验收。

---

## 1. 目录结构

```
project/
├── backend/                FastAPI 后端
│   ├── app/
│   │   ├── main.py         应用入口与路由装配
│   │   ├── config.py       配置（环境变量 MALFLOW_*）
│   │   ├── schemas.py      接口数据契约（前后端以此为准）
│   │   ├── routers/        health / traffic / tasks / models
│   │   └── services/       存储、推理封装、任务调度、模型注册
│   └── tests/              接口冒烟测试
├── frontend/               Vite + React + TypeScript + Tailwind
│   └── src/pages/          流量上传 / 结果展示 / 模型管理
├── ml/                     算法层
│   ├── features/           特征提取（extractor）与自适应选择（selection）
│   ├── data/               数据集构建与合成样本
│   ├── models/             MalFlowNet（CNN + BiLSTM + 门控统计特征）
│   ├── train.py            训练入口
│   └── predict.py          推理入口（后端调用）
├── environment/            requirements.txt / environment.yml
├── artifacts/              运行产物（上传文件、模型权重，不入库）
└── docs/                   仓库内文档索引
```

架构分层：前端三模块、后端三模块、ML 三模块、数据层三模块，与本目录一一对应。

---

## 2. 快速开始

### 2.1 克隆

```bash
git clone <仓库地址>
cd project
```

### 2.2 后端 + 算法层

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000
```

- 接口文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>

### 2.3 前端

```bash
cd frontend
npm install
npm run dev
```

访问 <http://localhost:5173>。开发服务器已把 `/api` 代理到 `127.0.0.1:8000`，
无需额外配置跨域。

### 2.4 训练一个可用权重（合成数据）

```bash
# 在项目根目录（project/）执行，需已激活虚拟环境
python -m ml.train --synthetic --epochs 8
```

产物写入 `artifacts/models/`，后端重启或在「模型管理」页点击激活后即切换为
真实模型推理，界面右上角标识由「演示推理模式」变为「已加载模型权重」。

数据集到位后改用：

```bash
python -m ml.train --data-dir data/raw --epochs 30
# data/raw/<类别名>/*.pcap，类别名见 ml/config.py 的 CLASS_NAMES
```

---

## 3. 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务与推理模式健康检查 |
| POST | `/api/traffic/upload` | 上传 pcap/pcapng/cap |
| GET | `/api/traffic/files` | 已上传文件列表 |
| POST | `/api/tasks` | 提交分类任务（异步，返回 task_id） |
| GET | `/api/tasks` | 任务列表 |
| GET | `/api/tasks/{task_id}` | 任务详情与分类结果 |
| GET | `/api/tasks/stats` | 仪表盘统计 |
| GET | `/api/models` | 模型列表 |
| GET | `/api/models/runtime` | 当前推理运行状态 |
| POST | `/api/models/{model_id}/activate` | 激活指定模型 |

---

## 4. 算法设计要点

| 需求条目 | 实现位置 |
|----------|----------|
| 流量特征分析与表示 | `ml/features/extractor.py`：40 维统计特征 + 包长方向序列 + 首部字节序列 |
| 字节特征 | 字节 Embedding + 1D-CNN；载荷熵、可打印字符比 |
| 包长与方向 | 带符号包长序列 + BiLSTM；包长分位数、突发统计 |
| 交互行为 | 到达间隔、方向切换率、上下行字节比、TLS 握手节奏 |
| 流间关联 | 同目的主机流数、时间窗并发度、同端口占比 |
| 特征贡献评估 | `ml/features/selection.py`：互信息 + 方差 + 树模型重要性加权融合 |
| 自适应选择 | 按累计贡献覆盖率动态取前 k 维，掩码写入模型作为门控先验 |
| 分类模型 | `ml/models/cnn_bilstm.py`：三路特征融合 MalFlowNet |
| 人机交互界面 | `frontend/src/pages/`：上传、结果展示、模型管理 |

---

## 5. 测试

```bash
cd backend && pytest -v            # 接口冒烟测试
python -m ml.predict --file <某个 pcap>   # 算法层单独自测（项目根目录执行）
cd frontend && npm run typecheck   # 前端类型检查
```

---

## 6. 环境与版本

- 目标环境：Ubuntu 22.04 / Python 3.10+ / Node.js 18 LTS
- 依赖版本见 `environment/requirements.txt`
- 环境验收：后端 `/api/health` 返回 200，前端可正常访问并完成上传与分类流程

**依赖降级策略**：`scapy` 缺失时特征提取退化为确定性伪流，`torch` 缺失或权重
未训练时推理退化为启发式基线。两种情况接口返回结构不变，仅 `mode` 字段不同，
以保证环境未就绪时演示与联调不被阻塞。

---

## 7. 协作规范

- 分支：`main`（可演示）/ `develop`（集成）/ `feature/*` / `fix/*`
- 提交格式：`<type>(<scope>): <subject>`，如 `feat(backend): add upload endpoint`

---

## 8. 版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1.0 | 2026-09-01 | 仓库初始化：前后端与算法层原型链路打通 |
