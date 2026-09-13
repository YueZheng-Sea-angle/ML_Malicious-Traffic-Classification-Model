"""用户注册、登录与个人资料接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import (
    AuthResponse,
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    UpdateProfileRequest,
    UserPublic,
)
from app.services.auth import (
    get_current_user,
    hash_password,
    security,
    user_to_public,
    verify_password,
)
from app.services.user_store import UserRecord, get_user_store

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_user_public(user: UserRecord) -> UserPublic:
    data = user_to_public(user)
    return UserPublic(**data)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> AuthResponse:
    store = get_user_store()
    salt, digest = hash_password(payload.password)
    try:
        user = store.create_user(
            username=payload.username.strip(),
            email=payload.email.strip(),
            password_salt=salt,
            password_hash=digest,
            display_name=payload.display_name.strip() if payload.display_name else None,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    token = store.create_session(user.user_id)
    return AuthResponse(access_token=token, user=_to_user_public(user))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    store = get_user_store()
    user = store.get_by_username(payload.username.strip())
    if user is None or not verify_password(payload.password, user.password_salt, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    token = store.create_session(user.user_id)
    return AuthResponse(access_token=token, user=_to_user_public(user))


@router.post("/logout")
def logout(
    user: UserRecord = Depends(get_current_user),
    credentials=Depends(security),
) -> dict:
    del user
    if credentials is not None:
        get_user_store().revoke_session(credentials.credentials)
    return {"ok": True}


@router.get("/me", response_model=UserPublic)
def me(user: UserRecord = Depends(get_current_user)) -> UserPublic:
    return _to_user_public(user)


@router.put("/profile", response_model=UserPublic)
def update_profile(
    payload: UpdateProfileRequest,
    user: UserRecord = Depends(get_current_user),
) -> UserPublic:
    store = get_user_store()
    fields = {}
    if payload.display_name is not None:
        fields["display_name"] = payload.display_name.strip()
    if payload.email is not None:
        fields["email"] = payload.email.strip()

    try:
        updated = store.update_user(user.user_id, **fields)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return _to_user_public(updated)


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    user: UserRecord = Depends(get_current_user),
) -> dict:
    if not verify_password(payload.current_password, user.password_salt, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="当前密码不正确")

    salt, digest = hash_password(payload.new_password)
    store = get_user_store()
    updated = store.update_user(user.user_id, password_salt=salt, password_hash=digest)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return {"ok": True}
