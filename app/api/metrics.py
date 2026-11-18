"""运维指标:从 ToolCallLog 聚合工具调用质量。"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import CurrentUser, SessionDep
from app.core.metrics import ToolCallSample, summarize_tool_calls
from app.db.models import ToolCallLog, utcnow

router = APIRouter(prefix="/api/metrics", tags=["ops"])

_MAX_SAMPLES = 10_000


class ToolMetricOut(BaseModel):
    tool_name: str
    calls: int
    failed_calls: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: int


class ToolMetricsOut(BaseModel):
    window_hours: int
    total_calls: int
    failed_calls: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: int
    tools: list[ToolMetricOut]
    generated_at: datetime


@router.get(
    "/tools",
    response_model=ToolMetricsOut,
    summary="工具调用指标",
    description=(
        "统计最近 N 小时内的工具调用次数、成功率、平均耗时与 P95 耗时。"
        "数据只来自 ToolCallLog,不会扫描工单或知识库等业务表。"
    ),
)
async def tool_metrics(
    user: CurrentUser,
    session: SessionDep,
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
    tool: Annotated[str | None, Query(max_length=64)] = None,
) -> ToolMetricsOut:
    since = utcnow() - timedelta(hours=hours)
    stmt = select(ToolCallLog.tool_name, ToolCallLog.ok, ToolCallLog.latency_ms).where(
        ToolCallLog.created_at >= since
    )
    if tool:
        stmt = stmt.where(ToolCallLog.tool_name == tool)

    # 限制样本量,避免数据量大时把整张表读进内存;报表仍是最近窗口的近似值。
    stmt = stmt.order_by(ToolCallLog.id.desc()).limit(_MAX_SAMPLES)
    rows = list(await session.execute(stmt))

    summary = summarize_tool_calls(
        ToolCallSample(tool_name=row.tool_name, ok=row.ok, latency_ms=row.latency_ms)
        for row in rows
    )
    return ToolMetricsOut(
        window_hours=hours,
        total_calls=summary.total_calls,
        failed_calls=summary.failed_calls,
        success_rate=summary.success_rate,
        avg_latency_ms=summary.avg_latency_ms,
        p95_latency_ms=summary.p95_latency_ms,
        tools=[
            ToolMetricOut(
                tool_name=item.tool_name,
                calls=item.calls,
                failed_calls=item.failed_calls,
                success_rate=item.success_rate,
                avg_latency_ms=item.avg_latency_ms,
                p95_latency_ms=item.p95_latency_ms,
            )
            for item in summary.tools
        ],
        generated_at=utcnow(),
    )
