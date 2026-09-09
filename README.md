# 加密代理工具识别系统（MalFlow）

> 基于深度学习方法、面向 DataCon2021-ETA 的加密代理/隧道流量**工具形态识别（T1，11 类）**

当前为 **v0.1.0 原型**：三层链路（前端 → 后端 → 算法）已打通。默认加载
`artifacts/models/malflow_datacon_tools.pt`（T1 权重），无可用权重时推理返回
`unavailable` 占位结果，不阻塞界面验收与联调。

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

**Linux / macOS（bash）**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000
```

**Windows（PowerShell）**

> 虚拟环境 `.venv` 位于**项目根目录**。请在项目根目录执行激活，不要先 `cd backend`；
> 若已在子目录，请退回项目根或改用 `..\.venv\Scripts\Activate.ps1`。

```powershell
cd <项目根目录>            # 回到项目根（克隆后即在此目录，勿先进 backend）

python -m venv .venv                            # 首次创建；已存在可跳过此步

# 首次激活如被“禁止运行脚本”拦截，先为本用户放行（一次性，需管理员身份或已授权）：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1                    # 激活成功时提示符前出现 (.venv)

$env:PYTHONUTF8 = 1                             # 必须：否则 pip 解析含中文注释的依赖清单会报 GBK 解码错
python -m pip install -r environment\requirements.txt

cd backend
python -m uvicorn app.main:app --reload --port 8000
```

> 常见报错“无法将 .\.venv\Scripts\Activate.ps1 识别为 cmdlet…”通常有两个原因：
> 1. 当前目录不在项目根（如在 `backend\` 下）——退回根目录或用 `..\.venv\Scripts\Activate.ps1`；
> 2. 执行策略禁止脚本——先运行上面的 `Set-ExecutionPolicy` 一行；或在 cmd 中改用 `activate.bat` 激活（路径 `.venv\Scripts\activate.bat`，同样须在项目根目录执行）。

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

### 2.4 训练/获取一个 T1 权重

仓库已在 `artifacts/models/malflow_datacon_tools.pt` 附带 T1（11 类代理工具）权重，
默认推理即使用它；界面右上角标识为「已加载模型权重」。

需要重训时（DataCon T1，需 `part1_label.txt` 存在）：

```bash
# 在项目根目录（project/）执行，需已激活虚拟环境
python -m ml.research.run_experiment --task tools --real-per-class 10 --epochs 8
```

`artifacts/models/` 下的权重可在「模型管理」页激活；若该目录没有任何 `.pt`，
推理返回 `unavailable` 占位结果（界面显示「模型未加载」）。

如自行按类别目录组织 PCAP（目录名取自 `ml/config.py` 的 11 类工具名），也可：

```bash
python -m ml.train --data-dir data/raw --epochs 30
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

**推理降级策略**：`scapy` 缺失或文件非法时特征提取退化为确定性伪流；`torch` 缺失
或 `artifacts/models/` 无权重时推理返回 `unavailable` 占位结果。两种情况接口返回
结构不变，仅 `mode` 字段不同，以保证环境未就绪时演示与联调不被阻塞。

---

## 7. 协作规范

- 分支：`main`（可演示）/ `develop`（集成）/ `feature/*` / `fix/*`
- 提交格式：`<type>(<scope>): <subject>`，如 `feat(backend): add upload endpoint`

---

## 8. 版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1.0 | 2026-09-01 | 仓库初始化：前后端与算法层原型链路打通 |
