"""会话历史接口:列表、消息、删除与跨用户隔离。

测试环境没有配置 DEEPSEEK_API_KEY,Agent 会在调用模型前就返回 error 事件,
但会话本身和本轮用户提问是在调用模型之前落库的,所以这里仍能验证
「一次对话 = 一条会话 + 一条用户消息」的持久化行为,而且全程不联网。
"""

import json
import uuid


def _parse_sse(text: str) -> list[dict]:
    """把 text/event-stream 响应体解析成 [{type, data}, ...],与前端逻辑一致。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        kind = "message"
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                kind = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append({"type": kind, "data": json.loads("\n".join(data_lines))})
    return events


def _chat(client, headers, message, conversation_id=None):
    payload = {"message": message}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = client.post("/api/chat", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    return _parse_sse(response.text)


def _conversation_id(events) -> int:
    return next(e["data"]["conversation_id"] for e in events if e["type"] == "meta")


def test_conversation_list_starts_empty(client, new_user):
    assert client.get("/api/conversations", headers=new_user["headers"]).json() == []


def test_chat_persists_conversation_and_message(client, new_user):
    events = _chat(client, new_user["headers"], "我有哪些工单?")
    conversation_id = _conversation_id(events)

    conversations = client.get("/api/conversations", headers=new_user["headers"]).json()
    assert len(conversations) == 1
    assert conversations[0]["id"] == conversation_id
    assert conversations[0]["title"] == "我有哪些工单?"
    assert conversations[0]["message_count"] == 1
    assert conversations[0]["last_message_at"] is not None

    messages = client.get(
        f"/api/conversations/{conversation_id}/messages", headers=new_user["headers"]
    ).json()
    assert [(m["role"], m["content"]) for m in messages] == [("user", "我有哪些工单?")]


def test_second_message_reuses_the_same_conversation(client, new_user):
    conversation_id = _conversation_id(_chat(client, new_user["headers"], "第一句"))

    _chat(client, new_user["headers"], "第二句", conversation_id=conversation_id)

    conversations = client.get("/api/conversations", headers=new_user["headers"]).json()
    assert len(conversations) == 1
    assert conversations[0]["message_count"] == 2

    messages = client.get(
        f"/api/conversations/{conversation_id}/messages", headers=new_user["headers"]
    ).json()
    assert [m["content"] for m in messages] == ["第一句", "第二句"]


def test_delete_conversation_removes_its_messages(client, new_user):
    conversation_id = _conversation_id(_chat(client, new_user["headers"], "帮我建个工单"))

    deleted = client.delete(
        f"/api/conversations/{conversation_id}", headers=new_user["headers"]
    )
    assert deleted.status_code == 204

    assert client.get("/api/conversations", headers=new_user["headers"]).json() == []
    # 会话已删,消息接口应当按「不存在」处理,而不是返回空列表
    assert (
        client.get(
            f"/api/conversations/{conversation_id}/messages", headers=new_user["headers"]
        ).status_code
        == 404
    )


def test_conversations_are_isolated_between_users(client, new_user):
    conversation_id = _conversation_id(_chat(client, new_user["headers"], "私密问题"))

    other = client.post(
        "/api/auth/register",
        json={"email": f"other{uuid.uuid4().hex[:8]}@example.com", "password": "password-123"},
    ).json()
    other_token = client.post(
        "/api/auth/login",
        json={"email": other["email"], "password": "password-123"},
    ).json()["access_token"]
    other_headers = {"Authorization": f"Bearer {other_token}"}

    assert client.get("/api/conversations", headers=other_headers).json() == []
    assert (
        client.get(
            f"/api/conversations/{conversation_id}/messages", headers=other_headers
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/api/conversations/{conversation_id}", headers=other_headers
        ).status_code
        == 404
    )


def test_unknown_conversation_returns_404(client, new_user):
    assert (
        client.get("/api/conversations/999999/messages", headers=new_user["headers"]).status_code
        == 404
    )
