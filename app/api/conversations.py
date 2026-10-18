"""会话历史:给前端加载会话列表和聊天记录用。"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import delete, func, select

from app.api.deps import CurrentUser, SessionDep
from app.db.models import Conversation, Message, ToolCallLog
from app.schemas.conversation import ConversationOut, MessageOut

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


async def _get_owned_conversation(conversation_id: int, user_id: int, session) -> Conversation:
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == user_id
        )
    )
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
    return conversation


@router.get(
    "",
    response_model=list[ConversationOut],
    summary="查询会话列表",
    description="只返回当前用户的会话,按最近一条消息的时间倒序。",
)
async def list_conversations(
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ConversationOut]:
    # 用一次 join + group by 把消息数和最后活跃时间带出来,避免每条会话再查一次
    activity = func.coalesce(func.max(Message.created_at), Conversation.created_at)
    stmt = (
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            func.count(Message.id),
            func.max(Message.created_at),
        )
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .where(Conversation.owner_id == user.id)
        .group_by(Conversation.id, Conversation.title, Conversation.created_at)
        .order_by(activity.desc(), Conversation.id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = await session.execute(stmt)
    return [
        ConversationOut(
            id=row[0],
            title=row[1],
            created_at=row[2],
            message_count=row[3],
            last_message_at=row[4],
        )
        for row in rows.all()
    ]


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageOut],
    summary="查询会话消息",
    description="按时间正序返回该会话的全部消息,供前端恢复聊天记录。",
)
async def list_messages(
    conversation_id: int,
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[Message]:
    conversation = await _get_owned_conversation(conversation_id, user.id, session)
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.id.asc())
        .limit(limit)
    )
    return list(await session.scalars(stmt))


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话",
    description="删除会话、会话下的消息以及工具调用轨迹,只能删自己的会话。",
)
async def delete_conversation(
    conversation_id: int, user: CurrentUser, session: SessionDep
) -> None:
    conversation = await _get_owned_conversation(conversation_id, user.id, session)
    # 工具调用轨迹没有定义 ORM 关系,这里显式清理,避免留下悬空的外键
    await session.execute(
        delete(ToolCallLog).where(ToolCallLog.conversation_id == conversation.id)
    )
    await session.delete(conversation)
    await session.commit()
