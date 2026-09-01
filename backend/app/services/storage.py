"""内存态存储。

原型阶段用进程内字典承载文件与任务记录，接口与后续替换为
SQLite/PostgreSQL 的 Repository 保持一致，切换时只改本文件。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FileRecord:
    file_id: str
    filename: str
    path: Path
    size_bytes: int
    uploaded_at: datetime


@dataclass
class TaskRecord:
    task_id: str
    file_id: str
    filename: str
    status: str = "pending"
    model_id: Optional[str] = None
    max_flows: int = 64
    note: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None
    result: Optional[dict] = None


class Store:
    def __init__(self) -> None:
        self._files: Dict[str, FileRecord] = {}
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    # ---------------- 文件 ---------------- #
    def add_file(self, filename: str, path: Path, size_bytes: int) -> FileRecord:
        record = FileRecord(
            file_id=uuid.uuid4().hex[:12],
            filename=filename,
            path=path,
            size_bytes=size_bytes,
            uploaded_at=datetime.now(),
        )
        with self._lock:
            self._files[record.file_id] = record
        return record

    def get_file(self, file_id: str) -> Optional[FileRecord]:
        return self._files.get(file_id)

    def list_files(self) -> List[FileRecord]:
        return sorted(self._files.values(), key=lambda r: r.uploaded_at, reverse=True)

    # ---------------- 任务 ---------------- #
    def add_task(self, file_record: FileRecord, model_id: Optional[str], max_flows: int,
                 note: Optional[str] = None) -> TaskRecord:
        task = TaskRecord(
            task_id=uuid.uuid4().hex[:12],
            file_id=file_record.file_id,
            filename=file_record.filename,
            model_id=model_id,
            max_flows=max_flows,
            note=note,
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 50, status: Optional[str] = None) -> List[TaskRecord]:
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks[:limit]

    def update_task(self, task_id: str, **fields) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in fields.items():
                setattr(task, key, value)
            return task

    def clear(self) -> None:
        with self._lock:
            self._files.clear()
            self._tasks.clear()


_store = Store()


def get_store() -> Store:
    return _store
