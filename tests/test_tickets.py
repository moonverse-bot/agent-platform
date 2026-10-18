"""工单接口:CRUD、过滤、统计与数据隔离。"""

import uuid


def _create(client, headers, **overrides):
    payload = {"title": "无法登录后台", "description": "输入正确密码仍提示失败"}
    payload.update(overrides)
    response = client.post("/api/tickets", json=payload, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_and_fetch_ticket(client, new_user):
    created = _create(client, new_user["headers"], priority="high")
    assert created["status"] == "open"
    assert created["priority"] == "high"

    fetched = client.get(f"/api/tickets/{created['id']}", headers=new_user["headers"])
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "无法登录后台"


def test_list_is_paginated_and_filterable(client, new_user):
    _create(client, new_user["headers"], title="第一条")
    closed = _create(client, new_user["headers"], title="第二条")
    client.patch(
        f"/api/tickets/{closed['id']}",
        json={"status": "closed"},
        headers=new_user["headers"],
    )

    all_tickets = client.get("/api/tickets", headers=new_user["headers"]).json()
    assert len(all_tickets) == 2

    open_only = client.get(
        "/api/tickets", params={"status": "open"}, headers=new_user["headers"]
    ).json()
    assert [t["title"] for t in open_only] == ["第一条"]

    limited = client.get(
        "/api/tickets", params={"limit": 1}, headers=new_user["headers"]
    ).json()
    assert len(limited) == 1


def test_stats_groups_by_status(client, new_user):
    first = _create(client, new_user["headers"])
    _create(client, new_user["headers"])
    client.patch(
        f"/api/tickets/{first['id']}", json={"status": "processing"}, headers=new_user["headers"]
    )

    stats = client.get("/api/tickets/stats", headers=new_user["headers"]).json()
    assert stats == {"open": 1, "processing": 1, "closed": 0, "total": 2}


def test_tickets_are_isolated_between_users(client, new_user):
    created = _create(client, new_user["headers"])

    other = client.post(
        "/api/auth/register",
        json={"email": f"other{uuid.uuid4().hex[:8]}@example.com", "password": "password-123"},
    ).json()
    other_token = client.post(
        "/api/auth/login",
        json={"email": other["email"], "password": "password-123"},
    ).json()["access_token"]
    other_headers = {"Authorization": f"Bearer {other_token}"}

    assert client.get(f"/api/tickets/{created['id']}", headers=other_headers).status_code == 404
    assert client.get("/api/tickets", headers=other_headers).json() == []


def test_invalid_status_is_rejected(client, new_user):
    response = client.get(
        "/api/tickets", params={"status": "unknown"}, headers=new_user["headers"]
    )
    assert response.status_code == 422
