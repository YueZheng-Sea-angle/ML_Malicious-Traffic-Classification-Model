"""接口数据契约（前后端联调以本文件为准）。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    inference_mode: str = Field(description="model=已加载权重，heuristic=演示推理")
    torch_available: bool


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime


class TaskCreateRequest(BaseModel):
    file_id: str
    model_id: Optional[str] = None
    max_flows: int = Field(default=64, ge=1, le=512)
    note: Optional[str] = None


class FeatureContribution(BaseModel):
    name: str
    weight: float
    value: float


class FlowResult(BaseModel):
    flow_id: str
    label: str
    label_zh: str
    confidence: float
    probabilities: Dict[str, float]
    meta: Dict[str, object] = {}


class TaskResult(BaseModel):
    label: str
    label_zh: str
    confidence: float
    malicious_score: float = 0.0
    probabilities: Dict[str, float]
    flow_count: int
    flows: List[FlowResult] = []
    top_features: List[FeatureContribution] = []
    mode: str


class TaskResponse(BaseModel):
    task_id: str
    file_id: str
    filename: str
    status: str = Field(description="pending / running / succeeded / failed")
    model_id: Optional[str] = None
    created_at: datetime
    finished_at: Optional[datetime] = None
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None
    result: Optional[TaskResult] = None


class TaskListResponse(BaseModel):
    total: int
    items: List[TaskResponse]


class ModelInfo(BaseModel):
    model_id: str
    name: str
    version: str
    framework: str = "PyTorch"
    is_active: bool = False
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    trained_at: Optional[datetime] = None
    selected_features: List[str] = []
    description: str = ""


class ModelListResponse(BaseModel):
    total: int
    active_model_id: Optional[str]
    items: List[ModelInfo]


class StatsResponse(BaseModel):
    total_tasks: int
    succeeded: int
    failed: int
    running: int
    label_distribution: Dict[str, int]
    average_elapsed_ms: float
