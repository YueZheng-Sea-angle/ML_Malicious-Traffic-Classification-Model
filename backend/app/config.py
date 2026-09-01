"""后端配置。

环境变量前缀 MALFLOW_，可通过 project/.env 覆盖（参考 .env.example）。
同时把项目根目录加入 sys.path，使后端可以直接 ``import ml``。
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Settings:
    """轻量配置对象，避免为了几个字段引入额外依赖层。"""

    def __init__(self) -> None:
        self.app_name: str = os.getenv("MALFLOW_APP_NAME", "恶意流量分类系统")
        self.version: str = "0.1.0"
        self.upload_dir: Path = _resolve(os.getenv("MALFLOW_UPLOAD_DIR", "artifacts/uploads"))
        self.model_dir: Path = _resolve(os.getenv("MALFLOW_MODEL_DIR", "artifacts/models"))
        self.max_upload_mb: int = int(os.getenv("MALFLOW_MAX_UPLOAD_MB", "200"))
        self.cors_origins: List[str] = [
            origin.strip()
            for origin in os.getenv(
                "MALFLOW_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if origin.strip()
        ]
        self.allowed_suffixes = {".pcap", ".pcapng", ".cap"}
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
