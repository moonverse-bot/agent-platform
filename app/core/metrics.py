"""工具调用指标的纯函数聚合。

把聚合逻辑从 API 层拆出来,一方面方便单测,另一方面后续做 CLI 报表或
定时任务时可以直接复用,不需要再复制一遍统计口径。
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from statistics import fmean


@dataclass(frozen=True)
class ToolCallSample:
    """一条工具调用样本,对应 ToolCallLog 里的一行。"""

    tool_name: str
    ok: bool
    latency_ms: int


@dataclass(frozen=True)
class ToolStats:
    tool_name: str
    calls: int
    failed_calls: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: int


@dataclass(frozen=True)
class ToolMetricsSummary:
    total_calls: int
    failed_calls: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: int
    tools: list[ToolStats]


def _p95(values: list[int]) -> int:
    """取 P95 的最近秩,样本很少时也不会抛异常。"""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _rate(ok_count: int, total: int) -> float:
    return round(ok_count / total, 4) if total else 0.0


def _avg(values: list[int]) -> float:
    return round(fmean(values), 1) if values else 0.0


def summarize_tool_calls(samples: Iterable[ToolCallSample]) -> ToolMetricsSummary:
    """按工具名聚合调用次数、成功率、平均耗时与 P95。"""

    grouped: dict[str, list[ToolCallSample]] = defaultdict(list)
    all_samples = list(samples)
    for sample in all_samples:
        grouped[sample.tool_name].append(sample)

    tools: list[ToolStats] = []
    for tool_name, items in grouped.items():
        latencies = [item.latency_ms for item in items]
        ok_count = sum(1 for item in items if item.ok)
        tools.append(
            ToolStats(
                tool_name=tool_name,
                calls=len(items),
                failed_calls=len(items) - ok_count,
                success_rate=_rate(ok_count, len(items)),
                avg_latency_ms=_avg(latencies),
                p95_latency_ms=_p95(latencies),
            )
        )

    # 调用量高的工具排前面;同调用量时按名字排序,保证结果稳定可比较。
    tools.sort(key=lambda item: (-item.calls, item.tool_name))

    all_latencies = [item.latency_ms for item in all_samples]
    ok_count = sum(1 for item in all_samples if item.ok)
    return ToolMetricsSummary(
        total_calls=len(all_samples),
        failed_calls=len(all_samples) - ok_count,
        success_rate=_rate(ok_count, len(all_samples)),
        avg_latency_ms=_avg(all_latencies),
        p95_latency_ms=_p95(all_latencies),
        tools=tools,
    )
