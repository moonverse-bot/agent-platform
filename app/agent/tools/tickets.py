"""工单相关工具:让 Agent 能直接查、能直接建。"""

import json

from sqlalchemy import select

from app.agent.tools.base import Tool, ToolContext
from app.db.models import Ticket

_STATUS_LABELS = {"open": "待处理", "processing": "处理中", "closed": "已关闭"}


def _serialize(ticket: Ticket) -> dict:
    return {
        "id": ticket.id,
        "title": ticket.title,
        "status": ticket.status,
        "priority": ticket.priority,
        "description": ticket.description[:200],
    }


async def create_ticket(ctx: ToolContext, args: dict) -> str:
    title = str(args.get("title", "")).strip()
    if not title:
        return "创建失败:title 不能为空。"

    ticket = Ticket(
        owner_id=ctx.user_id,
        title=title[:200],
        description=str(args.get("description", "")).strip()[:4000],
        priority=args.get("priority") or "normal",
    )
    ctx.session.add(ticket)
    await ctx.session.commit()
    await ctx.session.refresh(ticket)
    return json.dumps(
        {"created": _serialize(ticket), "message": f"工单已创建,编号 {ticket.id}"},
        ensure_ascii=False,
    )


async def list_my_tickets(ctx: ToolContext, args: dict) -> str:
    stmt = select(Ticket).where(Ticket.owner_id == ctx.user_id)
    status = args.get("status")
    if status:
        stmt = stmt.where(Ticket.status == status)
    stmt = stmt.order_by(Ticket.created_at.desc()).limit(int(args.get("limit") or 10))
    tickets = list(await ctx.session.scalars(stmt))
    if not tickets:
        return "当前没有符合条件的工单。"
    return json.dumps(
        {"count": len(tickets), "tickets": [_serialize(t) for t in tickets]},
        ensure_ascii=False,
    )


async def get_ticket(ctx: ToolContext, args: dict) -> str:
    try:
        ticket_id = int(args["ticket_id"])
    except (KeyError, TypeError, ValueError):
        return "查询失败:ticket_id 必须是整数。"

    ticket = await ctx.session.scalar(
        select(Ticket).where(Ticket.id == ticket_id, Ticket.owner_id == ctx.user_id)
    )
    if ticket is None:
        return f"未找到编号为 {ticket_id} 的工单,或它不属于当前用户。"

    payload = _serialize(ticket)
    payload["description"] = ticket.description
    payload["status_label"] = _STATUS_LABELS.get(ticket.status, ticket.status)
    return json.dumps(payload, ensure_ascii=False)


TICKET_TOOLS = [
    Tool(
        name="create_ticket",
        description="为用户创建一条新的工单,返回工单编号。当用户描述了一个需要人工跟进的问题时调用。",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "工单标题,一句话概括问题"},
                "description": {"type": "string", "description": "问题详情"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "优先级,默认 normal",
                },
            },
            "required": ["title"],
        },
        handler=create_ticket,
        tags=["ticket"],
    ),
    Tool(
        name="list_my_tickets",
        description="列出当前用户的工单,可按状态过滤。用户问「我的工单」「有哪些待处理的」时调用。",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "processing", "closed"],
                    "description": "可选,按状态过滤",
                },
                "limit": {"type": "integer", "description": "返回条数,默认 10"},
            },
        },
        handler=list_my_tickets,
        tags=["ticket"],
    ),
    Tool(
        name="get_ticket",
        description="按编号查询单条工单的详情,包括完整描述与状态。",
        parameters={
            "type": "object",
            "properties": {"ticket_id": {"type": "integer", "description": "工单编号"}},
            "required": ["ticket_id"],
        },
        handler=get_ticket,
        tags=["ticket"],
    ),
]
