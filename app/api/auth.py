"""注册、登录、查询当前用户。"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, SessionDep
from app.config import get_settings
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册账号",
    description="用邮箱和密码创建账号。邮箱需唯一,密码至少 8 位。",
)
async def register(payload: RegisterRequest, session: SessionDep) -> User:
    email = payload.email.strip().lower()
    if await session.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "该邮箱已注册")

    user = User(
        email=email,
        display_name=payload.display_name.strip() or email.split("@")[0],
        hashed_password=hash_password(payload.password),
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="登录获取令牌",
    description=(
        "校验邮箱和密码,成功后返回 JWT。后续需要鉴权的接口都要在请求头带上 "
        "`Authorization: Bearer <access_token>`。\n\n"
        "密码错误与账号不存在会返回同样的提示,避免被用来枚举账号。"
    ),
)
async def login(payload: LoginRequest, session: SessionDep) -> TokenResponse:
    email = payload.email.strip().lower()
    user = await session.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(payload.password, user.hashed_password):
        # 刻意不区分「用户不存在」和「密码错误」,避免账号枚举
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "邮箱或密码错误")

    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.email),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserOut, summary="查询当前登录用户")
async def me(user: CurrentUser) -> User:
    return user
