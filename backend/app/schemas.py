"""接口数据契约（前后端联调以本文件为准）。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    email: Optional[EmailStr] = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


class UserPublic(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    display_name: str
    role: str
    created_at: datetime
    updated_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    inference_mode: str = Field(description="model=已加载权重，unavailable=无可用权重")
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


class EvalSubmitRequest(BaseModel):
    model_id: Optional[str] = Field(default=None, description="缺省时用当前激活模型")
    train_ratio: float = Field(default=0.2, ge=0.05, le=0.95,
                               description="训练集比例（0.05-0.95），其余为测试集")
    per_class_files: int = Field(default=10, ge=1, le=100,
                                 description="每类从 real_data 抽取的文件数上限")


class EvalJobResponse(BaseModel):
    job_id: str
    model_id: str
    variant: str = Field(description="A=统计基线 / B=统计+n-gram")
    train_ratio: float
    per_class_files: int
    status: str = Field(description="running / succeeded / failed")
    error: Optional[str] = None
    elapsed_ms: Optional[int] = None
    result: Optional[Dict[str, object]] = None
