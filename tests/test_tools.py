"""工具层:直接对数据库操作,验证降级与数据隔离。"""

import json

from sqlalchemy import select

from app.agent.tools import build_default_registry
from app.agent.tools.base import ToolContext
from app.db.models import Ticket, ToolCallLog, User


async def test_create_ticket_tool_persists_and_logs(db_user: ToolContext):
    registry = build_default_registry()
    result = await registry.execute(
        "create_ticket", {"title": "登录报错 500", "priority": "urgent"}, db_user
    )

    assert result.ok is True
    payload = json.loads(result.output)
    assert payload["created"]["priority"] == "urgent"

    stored = await db_user.session.scalar(
        select(Ticket).where(Ticket.id == payload["created"]["id"])
    )
    assert stored is not None and stored.owner_id == db_user.user_id

    logs = list(await db_user.session.scalars(select(ToolCallLog)))
    assert any(log.tool_name == "create_ticket" and log.ok for log in logs)


async def test_get_ticket_tool_hides_other_users_tickets(db_user: ToolContext):
    stranger = User(
        email="stranger@example.com", display_name="陌生人", hashed_password="x"
    )
    db_user.session.add(stranger)
    await db_user.session.commit()
    await db_user.session.refresh(stranger)

    foreign = Ticket(owner_id=stranger.id, title="别人的工单")
    db_user.session.add(foreign)
    await db_user.session.commit()
    await db_user.session.refresh(foreign)

    result = await build_default_registry().execute(
        "get_ticket", {"ticket_id": foreign.id}, db_user
    )
    assert "未找到" in result.output


async def test_search_knowledge_tool_returns_relevant_article(db_user: ToolContext):
    result = await build_default_registry().execute(
        "search_knowledge", {"query": "退款要多久到账"}, db_user
    )
    assert result.ok is True
    payload = json.loads(result.output)
    assert payload["results"]
    assert any("退款" in item["title"] for item in payload["results"])


async def test_search_knowledge_tool_reports_empty_result(db_user: ToolContext):
    result = await build_default_registry().execute(
        "search_knowledge", {"query": "zzzz-不存在的主题-qqqq"}, db_user
    )
    assert "没有找到" in result.output


async def test_unknown_tool_returns_available_names(db_user: ToolContext):
    result = await build_default_registry().execute("drop_database", {}, db_user)
    assert result.ok is False
    assert "不存在" in result.output
    assert "create_ticket" in result.output


async def test_tool_exception_is_converted_to_text(db_user: ToolContext):
    from app.agent.tools.base import Tool, ToolRegistry

    async def boom(ctx, args):
        raise RuntimeError("下游服务超时")

    registry = ToolRegistry(
        [
            Tool(
                name="boom",
                description="总是失败的工具",
                parameters={"type": "object", "properties": {}},
                handler=boom,
            )
        ]
    )
    result = await registry.execute("boom", {}, db_user)
    assert result.ok is False
    assert "下游服务超时" in result.output
