"""数据库引擎与会话工厂。"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()

_engine_kwargs: dict = {}
if settings.is_sqlite:
    # SQLite 下用 NullPool:避免连接跨事件循环复用,也绕开测试里的锁表问题
    _engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(settings.database_url, **_engine_kwargs)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def create_all() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
