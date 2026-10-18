"""种子数据:演示账号只建一次,并且真的能用登录页预填的信息登进去。"""

from app.db.init_db import DEMO_EMAIL, DEMO_PASSWORD, seed_demo_user


async def test_demo_account_is_created_once_and_can_log_in(client):
    # client 夹具已经把库建好了,这里只验证幂等与可登录
    assert await seed_demo_user() is True
    assert await seed_demo_user() is False

    response = client.post(
        "/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD}
    )
    assert response.status_code == 200, response.text
    assert response.json()["access_token"]


async def test_demo_account_has_sample_data(client):
    """演示账号要有工单和一段历史会话,否则前端首屏还是空的。"""
    await seed_demo_user()
    token = client.post(
        "/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD}
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    tickets = client.get("/api/tickets", headers=headers).json()
    assert len(tickets) == 3
    assert {t["status"] for t in tickets} == {"open", "processing", "closed"}

    conversations = client.get("/api/conversations", headers=headers).json()
    assert len(conversations) == 1
    assert conversations[0]["message_count"] == 4
    assert conversations[0]["title"] == "退款政策是什么?"
