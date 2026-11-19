"""工具层:直接对数据库操作,验证降级与数据隔离。"""

import asyncio
import json
from datetime import timedelta

from sqlalchemy import select

from app.agent.tools import build_default_registry
from app.agent.tools.base import ToolContext
from app.db.models import Ticket, ToolCallLog, User, utcnow


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


async def test_create_ticket_tool_deduplicates_recent_identical_submission(
    db_user: ToolContext,
):
    registry = build_default_registry()
    arguments = {
        "title": "支付成功但订单未更新",
        "description": "订单 1001 支付成功后仍显示待支付。",
    }

    first = await registry.execute("create_ticket", arguments, db_user)
    second = await registry.execute("create_ticket", arguments, db_user)

    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first.ok is True and second.ok is True
    assert first_payload["duplicate"] is False
    assert second_payload["duplicate"] is True
    assert second_payload["created"]["id"] == first_payload["created"]["id"]

    different = await registry.execute(
        "create_ticket",
        {**arguments, "description": "同一标题,但补充了不同的故障现象。"},
        db_user,
    )
    different_payload = json.loads(different.output)
    assert different_payload["duplicate"] is False
    assert different_payload["created"]["id"] != first_payload["created"]["id"]

    tickets = list(
        await db_user.session.scalars(
            select(Ticket).where(Ticket.owner_id == db_user.user_id)
        )
    )
    assert len(tickets) == 2


async def test_create_ticket_tool_allows_repeat_after_dedupe_window(
    db_user: ToolContext,
):
    old_ticket = Ticket(
        owner_id=db_user.user_id,
        title="登录异常",
        description="密码正确但无法登录。",
        created_at=utcnow() - timedelta(minutes=10),
    )
    db_user.session.add(old_ticket)
    await db_user.session.commit()

    result = await build_default_registry().execute(
        "create_ticket",
        {"title": "登录异常", "description": "密码正确但无法登录。"},
        db_user,
    )
    payload = json.loads(result.output)

    assert payload["duplicate"] is False
    assert payload["created"]["id"] != old_ticket.id


async def test_create_ticket_tool_does_not_reuse_closed_ticket(db_user: ToolContext):
    closed_ticket = Ticket(
        owner_id=db_user.user_id,
        title="发票未收到",
        description="申请后一直没收到电子发票。",
        status="closed",
    )
    db_user.session.add(closed_ticket)
    await db_user.session.commit()

    result = await build_default_registry().execute(
        "create_ticket",
        {"title": "发票未收到", "description": "申请后一直没收到电子发票。"},
        db_user,
    )
    payload = json.loads(result.output)

    assert payload["duplicate"] is False
    assert payload["created"]["id"] != closed_ticket.id


async def test_create_ticket_tool_rejects_invalid_priority(db_user: ToolContext):
    result = await build_default_registry().execute(
        "create_ticket", {"title": "测试工单", "priority": "紧急"}, db_user
    )

    assert result.ok is False
    assert result.error == "invalid_arguments"
    assert "priority" in result.output

    tickets = list(
        await db_user.session.scalars(
            select(Ticket).where(Ticket.owner_id == db_user.user_id)
        )
    )
    assert tickets == []


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


async def test_get_ticket_tool_rejects_non_positive_id(db_user: ToolContext):
    result = await build_default_registry().execute(
        "get_ticket", {"ticket_id": 0}, db_user
    )

    assert result.ok is False
    assert result.error == "invalid_arguments"
    assert "ticket_id" in result.output


async def test_list_my_tickets_tool_rejects_invalid_limit(db_user: ToolContext):
    result = await build_default_registry().execute(
        "list_my_tickets", {"limit": "abc"}, db_user
    )

    assert result.ok is False
    assert result.error == "invalid_arguments"
    assert "limit" in result.output


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


async def test_search_knowledge_tool_rejects_blank_query(db_user: ToolContext):
    result = await build_default_registry().execute(
        "search_knowledge", {"query": "   "}, db_user
    )

    assert result.ok is False
    assert result.error == "invalid_arguments"
    assert "query" in result.output


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


async def test_tool_timeout_is_reported_and_logged(db_user: ToolContext):
    from app.agent.tools.base import Tool, ToolRegistry

    async def slow(ctx, args):
        await asyncio.sleep(0.05)
        return "迟到的结果"

    registry = ToolRegistry(
        [
            Tool(
                name="slow",
                description="总是超时的工具",
                parameters={"type": "object", "properties": {}},
                handler=slow,
            )
        ],
        default_timeout_seconds=0.01,
    )

    result = await registry.execute("slow", {}, db_user)

    assert result.ok is False
    assert result.error == "tool_timeout"
    assert "执行超时" in result.output

    logs = list(
        await db_user.session.scalars(
            select(ToolCallLog).where(ToolCallLog.tool_name == "slow")
        )
    )
    assert len(logs) == 1
    assert logs[0].ok is False
    assert "执行超时" in logs[0].result_preview


async def test_tool_timeout_can_be_overridden_per_tool(db_user: ToolContext):
    from app.agent.tools.base import Tool, ToolRegistry

    async def quick(ctx, args):
        await asyncio.sleep(0.01)
        return "完成"

    registry = ToolRegistry(
        [
            Tool(
                name="quick",
                description="快速工具",
                parameters={"type": "object", "properties": {}},
                handler=quick,
                timeout_seconds=1.0,
            )
        ],
        default_timeout_seconds=0.001,
    )

    result = await registry.execute("quick", {}, db_user)

    assert result.ok is True
    assert result.output == "完成"
