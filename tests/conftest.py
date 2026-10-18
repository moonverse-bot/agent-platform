"""测试夹具:使用独立的 SQLite 文件库,不碰开发数据,也不联网。"""

import os
import pathlib
import sys
import tempfile
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP_DIR = pathlib.Path(tempfile.mkdtemp(prefix="agent-platform-test-"))

# 必须在导入 app 之前设置:app.db.base 在导入时就会读配置并创建 engine
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TMP_DIR / 'test.db').as_posix()}"
os.environ["REDIS_URL"] = ""
os.environ["JWT_SECRET"] = "test-only-secret-at-least-32-bytes-long"
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"
# 测试里不要灌演示账号,避免「列表为空」这类断言被种子数据污染
os.environ["SEED_DEMO_DATA"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """带生命周期的测试客户端:会真正触发建表和种子数据。"""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def new_user(client) -> dict[str, str]:
    """注册并登录一个全新用户,返回请求体与鉴权头。"""
    email = f"u{uuid.uuid4().hex[:10]}@example.com"
    password = "password-123"

    registered = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "display_name": "测试用户"},
    )
    assert registered.status_code == 201, registered.text

    logged_in = client.post("/api/auth/login", json={"email": email, "password": password})
    assert logged_in.status_code == 200, logged_in.text
    token = logged_in.json()["access_token"]

    return {
        "email": email,
        "password": password,
        "user_id": str(registered.json()["id"]),
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest.fixture
async def db_user():
    """需要一个真实 User 行(以及一个可用 session)的工具/Agent 测试用。"""
    from app.agent.tools.base import ToolContext
    from app.core.security import hash_password
    from app.db.base import Base, SessionLocal, engine
    from app.db.init_db import seed_knowledge
    from app.db.models import User

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as session:
        await seed_knowledge()
        user = User(
            email=f"t{uuid.uuid4().hex[:10]}@example.com",
            display_name="工具测试用户",
            hashed_password=hash_password("password-123"),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        yield ToolContext(session=session, user_id=user.id, conversation_id=None)
