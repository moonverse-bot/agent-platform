"""对话接口:SSE 流式返回 Agent 的思考过程与回答。

事件类型:
  meta  - 会话信息
  tool  - 一次工具调用的入参与结果
  token - 回答的增量文本
  reset - 丢弃本轮已下发的 token(模型调工具前的铺垫文字)
  done  - 结束,附带完整回答
  error - 出错
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.agent.memory import load_history, save_message
from app.agent.runner import AgentRunner
from app.api.deps import AgentDep, CurrentUser, RateLimiterDep
from app.core.logging import get_trace_id
from app.db.base import SessionLocal
from app.db.models import Conversation
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _resolve_conversation(session, user_id: int, conversation_id: int | None, title: str) -> Conversation:
    if conversation_id is not None:
        conversation = await session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.owner_id == user_id
            )
        )
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在")
        return conversation

    conversation = Conversation(owner_id=user_id, title=title[:200] or "新会话")
    session.add(conversation)
    await session.flush()
    return conversation


@router.post(
    "",
    summary="与 Agent 对话(SSE 流式)",
    description=(
        "把用户消息交给 Agent,以 `text/event-stream` 流式返回。\n\n"
        "事件顺序:`meta`(会话信息)→ 若干 `tool`(工具调用)→ 若干 `token`(回答增量)→ `done`(结束)。"
        "若模型在调用工具前先输出了一段铺垫文字,会补一个 `reset` 事件通知前端丢弃那段文本。\n\n"
        "注意这里返回的是流,浏览器里直接打开看不到内容,建议用 curl 并加上 -N 参数。"
    ),
)
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    limiter: RateLimiterDep,
    agent: AgentDep,
) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "message 不能为空")

    conversation_id = payload.conversation_id

    # 限流放在真正开始推流之前,超限直接返回 429,避免白占用一次模型调用
    quota = await limiter.hit(f"user:{user.id}")
    if not quota.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "请求过于频繁,请稍后再试",
            headers={"Retry-After": "60"},
        )

    user_id = user.id
    trace_id = get_trace_id()

    async def event_stream():
        # 流式响应会持续到生成结束,这里另开一个会话,不依赖请求级依赖的生命周期
        async with SessionLocal() as session:
            try:
                conversation = await _resolve_conversation(session, user_id, conversation_id, message)
                await session.flush()
                # 先取历史再落库本轮提问,否则当前这句会被当成历史重复带进上下文
                history = await load_history(session, conversation.id)
                await save_message(session, conversation.id, "user", message)
                await session.commit()
            except HTTPException as exc:
                yield _sse("error", {"message": exc.detail})
                return

            yield _sse(
                "meta",
                {"conversation_id": conversation.id, "trace_id": trace_id, "remaining": quota.remaining},
            )

            final_text = ""
            error_text = ""

            async for event in agent.stream(
                session=session,
                user_id=user_id,
                conversation_id=conversation.id,
                history=history,
                user_text=message,
            ):
                if event.type == "error":
                    error_text = event.data.get("message", "未知错误")
                    yield _sse("error", event.data)
                elif event.type == "done":
                    final_text = event.data.get("content", "")
                else:
                    # token / tool / reset 直接透传
                    yield _sse(event.type, event.data)

            if final_text:
                await save_message(session, conversation.id, "assistant", final_text)
                await session.commit()

            yield _sse(
                "done",
                {
                    "conversation_id": conversation.id,
                    "content": final_text,
                    "error": error_text or None,
                    "remaining": quota.remaining,
                    "limit": quota.limit,
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
