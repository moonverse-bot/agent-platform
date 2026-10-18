"""FastAPI 依赖:数据库会话、当前用户、限流器、Agent 运行时。"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runner import AgentRunner
from app.core.ratelimit import RateLimiter
from app.core.security import decode_access_token
from app.db.base import get_session
from app.db.models import User

SessionDep = Annotated[AsyncSession, Depends(get_session)]

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "缺少 Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    email = (payload or {}).get("sub")
    if not email:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await session.scalar(select(User).where(User.email == email))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_agent(request: Request) -> AgentRunner:
    return request.app.state.agent


RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
AgentDep = Annotated[AgentRunner, Depends(get_agent)]
