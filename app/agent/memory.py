"""会话记忆:Phase 1 直接落库,按最近 N 轮回填上下文。"""

from sqlalchemy import select

from app.db.models import Message

DEFAULT_HISTORY_PAIRS = 10


async def load_history(
    session, conversation_id: int, limit_pairs: int = DEFAULT_HISTORY_PAIRS
) -> list[dict[str, str]]:
    """取最近 N 轮对话,按时间正序返回,可直接拼进 messages。"""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(limit_pairs * 2)
    )
    rows = list(await session.scalars(stmt))
    rows.reverse()
    return [{"role": row.role, "content": row.content} for row in rows if row.content]


async def save_message(session, conversation_id: int, role: str, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(message)
    await session.flush()
    return message
