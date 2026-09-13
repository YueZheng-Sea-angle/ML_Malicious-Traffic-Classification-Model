"""FastAPI 应用入口。

启动（开发）：
    cd project/backend && uvicorn app.main:app --reload --port 8000
接口文档：
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import auth, health, models, tasks, traffic

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="基于深度学习的加密代理/隧道流量工具识别系统 · 后端 API（DataCon T1 口径）",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(traffic.router, prefix=API_PREFIX)
app.include_router(tasks.router, prefix=API_PREFIX)
app.include_router(models.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
def index() -> JSONResponse:
    return JSONResponse(
        {
            "app": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "api_prefix": API_PREFIX,
        }
    )
