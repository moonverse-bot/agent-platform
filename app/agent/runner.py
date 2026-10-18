"""Agent 运行时:把模型、工具和会话记忆串成一个可流式的执行循环。

Phase 1 用显式循环实现「提问 → 选工具 → 执行 → 观察 → 作答」这条链路。
好处是没有隐藏魔法、每一步都能打断点;等流程变复杂(条件分支、并行工具、
人工介入、子图复用)再换成 LangGraph 做图编排。
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.agent.llm import build_client
from app.agent.tools import ToolRegistry, build_default_registry
from app.agent.tools.base import ToolContext, ToolResult
from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是「云雀」客服平台的智能助手,负责回答产品问题,并在需要时帮用户创建或查询工单。\n"
    "工作要求:\n"
    "1. 涉及产品政策、流程、费用的问题,先调用 search_knowledge 查资料再回答,不要凭记忆编造。\n"
    "2. 需要调用工具时直接调用,不要先输出「我这就去查」之类的铺垫文字。\n"
    "3. 用户明确要求提交问题、或问题需要人工跟进的,调用 create_ticket 建单,并把编号告诉用户。\n"
    "4. 用户询问自己的工单时,调用 list_my_tickets 或 get_ticket,只返回属于当前用户的数据。\n"
    "5. 只使用工具返回的信息作答;查不到就如实说明,并建议提交工单。回答用简洁的中文。"
)


@dataclass
class AgentEvent:
    """一次可流式下发的 Agent 事件。"""

    type: str  # token | tool | reset | done | error
    data: dict[str, Any]


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        client: Any = None,
        registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        self._settings = settings
        self._client = client if client is not None else build_client(settings)
        self._registry = registry or build_default_registry()
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._temperature = temperature

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def tool_names(self) -> list[str]:
        return self._registry.names()

    async def _create_stream(self, messages: list[dict], tool_specs: list[dict]):
        params: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": True,
        }
        if tool_specs:
            params["tools"] = tool_specs
            params["tool_choice"] = "auto"
        # 推理模型不接受 temperature,按模型决定是否下发,否则接口直接报 400
        if "reasoner" not in self._settings.llm_model:
            params["temperature"] = self._temperature
        return await self._client.chat.completions.create(**params)

    async def stream(
        self,
        *,
        session,
        user_id: int,
        conversation_id: int | None,
        history: list[dict[str, str]],
        user_text: str,
    ) -> AsyncIterator[AgentEvent]:
        if self._client is None:
            yield AgentEvent(
                "error",
                {"message": "未配置 DEEPSEEK_API_KEY,Agent 无法调用大模型。请在 .env 中补齐后重试。"},
            )
            return

        ctx = ToolContext(session=session, user_id=user_id, conversation_id=conversation_id)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        tool_specs = self._registry.specs()
        max_rounds = max(1, self._settings.llm_max_tool_rounds)

        for _round in range(max_rounds):
            try:
                stream = await self._create_stream(messages, tool_specs)
            except Exception as exc:  # noqa: BLE001 - 统一转成 error 事件下发
                logger.exception("调用大模型失败")
                yield AgentEvent("error", {"message": f"调用大模型失败:{exc}"})
                return

            content_parts: list[str] = []
            # index -> {id, name, arguments}:流式返回的 tool_calls 是按片段拼接的
            pending: dict[int, dict[str, str]] = {}

            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    if getattr(delta, "content", None):
                        content_parts.append(delta.content)
                        yield AgentEvent("token", {"text": delta.content})

                    for call in getattr(delta, "tool_calls", None) or []:
                        index = call.index if call.index is not None else 0
                        acc = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if getattr(call, "id", None):
                            acc["id"] = call.id
                        function = getattr(call, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                acc["name"] += function.name
                            if getattr(function, "arguments", None):
                                acc["arguments"] += function.arguments
            except Exception as exc:  # noqa: BLE001
                logger.exception("读取模型流失败")
                yield AgentEvent("error", {"message": f"读取模型流失败:{exc}"})
                return

            if not pending:
                yield AgentEvent("done", {"content": "".join(content_parts)})
                return

            if content_parts:
                # 模型在调工具前常会先吐一句「我这就去查」。它不是最终答案,
                # 但已经按 token 推给前端了,所以显式通知前端把这一段丢掉,
                # 保证界面上显示的与 done.content 落库的内容一致。
                yield AgentEvent(
                    "reset",
                    {"reason": "tool_call", "discarded": "".join(content_parts)},
                )

            ordered = sorted(pending.items())
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": acc["id"] or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": acc["name"],
                                "arguments": acc["arguments"] or "{}",
                            },
                        }
                        for index, acc in ordered
                    ],
                }
            )

            for index, acc in ordered:
                name = acc["name"]
                tool_call_id = acc["id"] or f"call_{index}"
                arguments = self._parse_arguments(acc["arguments"])
                if arguments is None:
                    result = ToolResult(
                        ok=False,
                        output=f"参数解析失败:工具 {name} 的参数不是合法 JSON 对象。",
                        latency_ms=0,
                        error="invalid_arguments",
                    )
                else:
                    result = await self._registry.execute(name, arguments, ctx)

                yield AgentEvent(
                    "tool",
                    {
                        "name": name,
                        "arguments": arguments or {},
                        "ok": result.ok,
                        "latency_ms": result.latency_ms,
                        "result": result.output,
                    },
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": result.output}
                )

        yield AgentEvent(
            "done",
            {"content": "本次处理超出最大工具调用轮数,已终止。请换一种问法或提交工单。"},
        )

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any] | None:
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
