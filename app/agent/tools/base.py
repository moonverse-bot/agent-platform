"""工具抽象:声明式描述 + 统一执行入口(含耗时统计与轨迹落库)。

工具是 Agent 触达真实业务的唯一通道,所以这里必须做三件事:
  1. 参数 schema 交给模型,由模型的 function calling 负责填参;
  2. 执行前后统一埋点,把参数、结果预览、耗时、成功与否写进 ToolCallLog;
  3. 任何异常都转成模型能读懂的文本,而不是让整条流断掉。
"""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ToolCallLog

logger = logging.getLogger(__name__)

_PREVIEW_LIMIT = 500


@dataclass
class ToolContext:
    session: AsyncSession
    user_id: int
    conversation_id: int | None = None


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    tags: list[str] = field(default_factory=list)

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    output: str
    latency_ms: int
    error: str | None = None


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(name)

        if tool is None:
            result = ToolResult(
                ok=False,
                output=f"错误:不存在名为 {name} 的工具。可用工具:{', '.join(self.names())}",
                latency_ms=0,
                error="unknown_tool",
            )
        else:
            try:
                output = await tool.handler(ctx, arguments)
                result = ToolResult(ok=True, output=output, latency_ms=0)
            except Exception as exc:  # noqa: BLE001 - 工具异常不应该打断整条对话流
                logger.exception("工具执行失败: %s", name)
                result = ToolResult(
                    ok=False,
                    output=f"工具 {name} 执行失败:{exc}",
                    latency_ms=0,
                    error=str(exc),
                )

        result.latency_ms = int((time.perf_counter() - started) * 1000)
        await self._record(ctx, name, arguments, result)
        return result

    async def _record(
        self,
        ctx: ToolContext,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> None:
        log = ToolCallLog(
            conversation_id=ctx.conversation_id,
            tool_name=name,
            arguments=json.dumps(arguments, ensure_ascii=False)[:_PREVIEW_LIMIT],
            result_preview=result.output[:_PREVIEW_LIMIT],
            ok=result.ok,
            latency_ms=result.latency_ms,
        )
        ctx.session.add(log)
        try:
            await ctx.session.commit()
        except Exception:  # noqa: BLE001 - 埋点失败不能影响主流程
            logger.exception("写入工具调用轨迹失败")
            await ctx.session.rollback()

        logger.info(
            "工具调用 %s",
            name,
            extra={"event": "agent.tool", "tool": name, "ok": result.ok,
                   "latency_ms": result.latency_ms},
        )
