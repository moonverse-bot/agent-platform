"""工单 CRUD:所有查询都按 owner_id 收敛,保证数据隔离。"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, SessionDep
from app.db.models import Ticket
from app.schemas.ticket import TicketCreate, TicketOut, TicketStatus, TicketUpdate

router = APIRouter(prefix="/api/tickets", tags=["tickets"])


async def _get_owned_ticket(ticket_id: int, user_id: int, session) -> Ticket:
    ticket = await session.scalar(
        select(Ticket).where(Ticket.id == ticket_id, Ticket.owner_id == user_id)
    )
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "工单不存在")
    return ticket


@router.post(
    "",
    response_model=TicketOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建工单",
    description="为当前登录用户创建一条工单,返回工单编号。",
)
async def create_ticket(
    payload: TicketCreate, user: CurrentUser, session: SessionDep
) -> Ticket:
    ticket = Ticket(
        owner_id=user.id,
        title=payload.title.strip(),
        description=payload.description.strip(),
        priority=payload.priority,
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    return ticket


@router.get(
    "",
    response_model=list[TicketOut],
    summary="查询工单列表",
    description="只返回当前用户的工单,支持按状态过滤与分页。",
)
async def list_tickets(
    user: CurrentUser,
    session: SessionDep,
    status_filter: Annotated[TicketStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Ticket]:
    stmt = select(Ticket).where(Ticket.owner_id == user.id)
    if status_filter:
        stmt = stmt.where(Ticket.status == status_filter)
    stmt = stmt.order_by(Ticket.created_at.desc(), Ticket.id.desc()).limit(limit).offset(offset)
    return list(await session.scalars(stmt))


@router.get(
    "/stats",
    summary="工单状态统计",
    description="按状态汇总当前用户的工单数量,额外返回一个 total。",
)
async def ticket_stats(user: CurrentUser, session: SessionDep) -> dict[str, int]:
    rows = await session.execute(
        select(Ticket.status, func.count())
        .where(Ticket.owner_id == user.id)
        .group_by(Ticket.status)
    )
    stats = {"open": 0, "processing": 0, "closed": 0}
    total = 0
    for ticket_status, count in rows.all():
        stats[ticket_status] = count
        total += count
    stats["total"] = total
    return stats


@router.get(
    "/{ticket_id}",
    response_model=TicketOut,
    summary="查询单条工单",
    description=(
        "按编号查询工单详情。若工单不属于当前用户,统一返回 404 而不是 403,"
        "避免通过状态码推测某个编号是否存在。"
    ),
)
async def get_ticket(ticket_id: int, user: CurrentUser, session: SessionDep) -> Ticket:
    return await _get_owned_ticket(ticket_id, user.id, session)


@router.patch(
    "/{ticket_id}",
    response_model=TicketOut,
    summary="更新工单",
    description="修改工单的状态或优先级,只传需要改的字段即可。",
)
async def update_ticket(
    ticket_id: int, payload: TicketUpdate, user: CurrentUser, session: SessionDep
) -> Ticket:
    ticket = await _get_owned_ticket(ticket_id, user.id, session)
    if payload.status is not None:
        ticket.status = payload.status
    if payload.priority is not None:
        ticket.priority = payload.priority
    await session.commit()
    await session.refresh(ticket)
    return ticket
