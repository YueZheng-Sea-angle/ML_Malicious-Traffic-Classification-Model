"""流量文件上传。"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import UploadResponse
from app.services.storage import get_store

router = APIRouter(prefix="/traffic", tags=["traffic"])

CHUNK_SIZE = 1024 * 1024


@router.post("/upload", response_model=UploadResponse, summary="上传流量文件")
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 {'/'.join(sorted(settings.allowed_suffixes))} 格式，收到 {suffix or '无扩展名'}",
        )

    store = get_store()
    target = settings.upload_dir / f"{_safe_stem(file.filename)}{suffix}"
    target = _dedupe(target)

    size = 0
    with target.open("wb") as handle:
        while chunk := await file.read(CHUNK_SIZE):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 上限"
                )
            handle.write(chunk)

    record = store.add_file(file.filename or target.name, target, size)
    return UploadResponse(
        file_id=record.file_id,
        filename=record.filename,
        size_bytes=record.size_bytes,
        uploaded_at=record.uploaded_at,
    )


@router.get("/files", response_model=List[UploadResponse], summary="已上传文件列表")
def list_files() -> List[UploadResponse]:
    return [
        UploadResponse(
            file_id=r.file_id,
            filename=r.filename,
            size_bytes=r.size_bytes,
            uploaded_at=r.uploaded_at,
        )
        for r in get_store().list_files()
    ]


def _safe_stem(filename: str | None) -> str:
    stem = Path(filename or "traffic").stem
    cleaned = "".join(c for c in stem if c.isalnum() or c in "-_")[:48]
    return cleaned or "traffic"


def _dedupe(path: Path) -> Path:
    candidate, index = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        index += 1
    return candidate
