"""用户与会话存储（JSON 持久化 + 内存会话）。

原型阶段用本地 JSON 文件保存账号，会话 token 保存在进程内字典。
后续可替换为 SQLite/PostgreSQL，接口保持不变。
"""

from __future__ import annotations

import json
import secrets
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings


@dataclass
class UserRecord:
    user_id: str
    username: str
    email: str
    display_name: str
    password_salt: str
    password_hash: str
    role: str = "user"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class UserStore:
    def __init__(self, data_path: Path) -> None:
        self._path = data_path
        self._users: Dict[str, UserRecord] = {}
        self._username_index: Dict[str, str] = {}
        self._email_index: Dict[str, str] = {}
        self._sessions: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self._ensure_demo_user()

    # ---------------- 用户 ---------------- #
    def create_user(
        self,
        username: str,
        email: str,
        password_salt: str,
        password_hash: str,
        display_name: Optional[str] = None,
    ) -> UserRecord:
        username_key = username.lower()
        email_key = email.lower()
        with self._lock:
            if username_key in self._username_index:
                raise ValueError("用户名已被占用")
            if email_key in self._email_index:
                raise ValueError("邮箱已被注册")

            user = UserRecord(
                user_id=uuid.uuid4().hex[:12],
                username=username,
                email=email,
                display_name=display_name or username,
                password_salt=password_salt,
                password_hash=password_hash,
            )
            self._users[user.user_id] = user
            self._username_index[username_key] = user.user_id
            self._email_index[email_key] = user.user_id
            self._persist_locked()
            return user

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        return self._users.get(user_id)

    def get_by_username(self, username: str) -> Optional[UserRecord]:
        user_id = self._username_index.get(username.lower())
        return self._users.get(user_id) if user_id else None

    def update_user(self, user_id: str, **fields) -> Optional[UserRecord]:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                return None

            if "email" in fields:
                email_key = str(fields["email"]).lower()
                existing = self._email_index.get(email_key)
                if existing and existing != user_id:
                    raise ValueError("邮箱已被注册")
                del self._email_index[user.email.lower()]
                user.email = str(fields["email"])
                self._email_index[email_key] = user_id

            if "display_name" in fields:
                user.display_name = str(fields["display_name"])

            if "password_salt" in fields:
                user.password_salt = str(fields["password_salt"])
            if "password_hash" in fields:
                user.password_hash = str(fields["password_hash"])

            user.updated_at = datetime.now()
            self._persist_locked()
            return user

    def list_users(self) -> List[UserRecord]:
        return sorted(self._users.values(), key=lambda u: u.created_at)

    # ---------------- 会话 ---------------- #
    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = user_id
        return token

    def resolve_session(self, token: str) -> Optional[UserRecord]:
        user_id = self._sessions.get(token)
        return self._users.get(user_id) if user_id else None

    def revoke_session(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def clear(self) -> None:
        with self._lock:
            self._users.clear()
            self._username_index.clear()
            self._email_index.clear()
            self._sessions.clear()
            if self._path.exists():
                self._path.unlink()

    # ---------------- 持久化 ---------------- #
    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for item in raw.get("users", []):
            user = UserRecord(
                user_id=item["user_id"],
                username=item["username"],
                email=item["email"],
                display_name=item["display_name"],
                password_salt=item["password_salt"],
                password_hash=item["password_hash"],
                role=item.get("role", "user"),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            self._users[user.user_id] = user
            self._username_index[user.username.lower()] = user.user_id
            self._email_index[user.email.lower()] = user.user_id

    def _persist_locked(self) -> None:
        payload = {
            "users": [
                {
                    **asdict(user),
                    "created_at": user.created_at.isoformat(),
                    "updated_at": user.updated_at.isoformat(),
                }
                for user in self._users.values()
            ]
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_demo_user(self) -> None:
        if self.get_by_username("demo"):
            return
        from app.services.auth import hash_password

        salt, digest = hash_password("demo123456")
        self.create_user(
            username="demo",
            email="demo@malflow.local",
            password_salt=salt,
            password_hash=digest,
            display_name="演示账号",
        )


_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = UserStore(settings.users_file)
    return _store
