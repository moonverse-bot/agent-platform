"""工单相关工具:让 Agent 能直接查、能直接建。"""

import json
from datetime import timedelta
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.agent.tools.base import Tool, ToolContext
from app.config import get_settings
from app.db.models import Ticket, utcnow

_STATUS_LABELS = {"open": "待处理", "processing": "处理中", "closed": "已关闭"}

Priority = Literal["low", "normal", "high", "urgent"]
TicketStatusArg = Literal["open", "processing", "closed"]
_PRIORITY_VALUES = get_args(Priority)
_STATUS_VALUES = get_args(TicketStatusArg)


class CreateTicketArgs(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    priority: Priority = "normal"

    @field_validator("title", "description", mode="before")
    @classmethod
    def normalize_text(cls, value):
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value):
        if value is None:
            return "normal"
        return str(value).strip().lower()


class ListMyTicketsArgs(BaseModel):
    status: TicketStatusArg | None = None
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value):
        if value is None or not str(value).strip():
            return None
        return str(value).strip().lower()

    @field_validator("limit", mode="before")
    @classmethod
    def normalize_limit(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return 10
        return value


class GetTicketArgs(BaseModel):
    ticket_id: int = Field(ge=1)


def _serialize(ticket: Ticket) -> dict:
    return {
        "id": ticket.id,
        "title": ticket.title,
        "status": ticket.status,
        "priority": ticket.priority,
        "description": ticket.description[:200],
    }


async def _find_recent_duplicate(
    ctx: ToolContext, title: str, description: str
) -> Ticket | None:
    window_seconds = get_settings().ticket_duplicate_window_seconds
    if window_seconds <= 0:
        return None

    cutoff = utcnow() - timedelta(seconds=window_seconds)
    stmt = (
        select(Ticket)
        .where(
            Ticket.owner_id == ctx.user_id,
            Ticket.title == title,
            Ticket.description == description,
            Ticket.status != "closed",
            Ticket.created_at >= cutoff,
        )
        .order_by(Ticket.id.desc())
        .limit(1)
    )
    return await ctx.session.scalar(stmt)


async def create_ticket(ctx: ToolContext, args: dict) -> str:
    title = args["title"]
    description = args.get("description", "")

    duplicate = await _find_recent_duplicate(ctx, title, description)
    if duplicate is not None:
        return json.dumps(
            {
                "created": _serialize(duplicate),
                "duplicate": True,
                "message": f"检测到重复提交,已返回原工单,编号 {duplicate.id}",
            },
            ensure_ascii=False,
        )

    ticket = Ticket(
        owner_id=ctx.user_id,
        title=title,
        description=description,
        priority=args["priority"],
    )
    ctx.session.add(ticket)
    await ctx.session.commit()
    await ctx.session.refresh(ticket)
    return json.dumps(
        {
            "created": _serialize(ticket),
            "duplicate": False,
            "message": f"工单已创建,编号 {ticket.id}",
        },
        ensure_ascii=False,
    )


async def list_my_tickets(ctx: ToolContext, args: dict) -> str:
    stmt = select(Ticket).where(Ticket.owner_id == ctx.user_id)
    status = args.get("status")
    if status:
        stmt = stmt.where(Ticket.status == status)
    stmt = stmt.order_by(Ticket.created_at.desc()).limit(args.get("limit", 10))
    tickets = list(await ctx.session.scalars(stmt))
    if not tickets:
        return "当前没有符合条件的工单。"
    return json.dumps(
        {"count": len(tickets), "tickets": [_serialize(t) for t in tickets]},
        ensure_ascii=False,
    )


async def get_ticket(ctx: ToolContext, args: dict) -> str:
    ticket_id = args["ticket_id"]

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
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "工单标题,一句话概括问题",
                },
                "description": {
                    "type": "string",
                    "maxLength": 4000,
                    "description": "问题详情",
                },
                "priority": {
                    "type": "string",
                    "enum": list(_PRIORITY_VALUES),
                    "description": "优先级,默认 normal",
                },
            },
            "required": ["title"],
        },
        handler=create_ticket,
        tags=["ticket"],
        args_model=CreateTicketArgs,
    ),
    Tool(
        name="list_my_tickets",
        description="列出当前用户的工单,可按状态过滤。用户问「我的工单」「有哪些待处理的」时调用。",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(_STATUS_VALUES),
                    "description": "可选,按状态过滤",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "返回条数,默认 10,最多 50",
                },
            },
        },
        handler=list_my_tickets,
        tags=["ticket"],
        args_model=ListMyTicketsArgs,
    ),
    Tool(
        name="get_ticket",
        description="按编号查询单条工单的详情,包括完整描述与状态。",
        parameters={
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "工单编号(正整数)",
                }
            },
            "required": ["ticket_id"],
        },
        handler=get_ticket,
        tags=["ticket"],
        args_model=GetTicketArgs,
    ),
]
