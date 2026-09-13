"""认证工具：密码哈希、Bearer Token 解析。"""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.user_store import UserRecord, get_user_store

security = HTTPBearer(auto_error=False)
_ITERATIONS = 100_000


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    ).hex()
    return salt, digest


def verify_password(password: str, salt: str, digest: str) -> bool:
    _, computed = hash_password(password, salt)
    return secrets.compare_digest(computed, digest)


def user_to_public(user: UserRecord) -> dict:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserRecord:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="请先登录")

    user = get_user_store().resolve_session(credentials.credentials)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录")
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[UserRecord]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    return get_user_store().resolve_session(credentials.credentials)
