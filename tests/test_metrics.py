"""工具调用指标:纯函数口径与接口鉴权/空数据。"""

from app.core.metrics import ToolCallSample, summarize_tool_calls


def test_summarize_tool_calls_groups_and_computes_p95():
    samples = [
        ToolCallSample("search_knowledge", True, 100),
        ToolCallSample("search_knowledge", True, 200),
        ToolCallSample("search_knowledge", False, 300),
        ToolCallSample("create_ticket", True, 50),
        ToolCallSample("create_ticket", True, 150),
    ]

    summary = summarize_tool_calls(samples)

    assert summary.total_calls == 5
    assert summary.failed_calls == 1
    assert summary.success_rate == 0.8

    search = next(item for item in summary.tools if item.tool_name == "search_knowledge")
    assert search.calls == 3
    assert search.failed_calls == 1
    assert search.success_rate == 0.6667
    assert search.avg_latency_ms == 200.0
    assert search.p95_latency_ms == 300

    create = next(item for item in summary.tools if item.tool_name == "create_ticket")
    assert create.calls == 2
    assert create.p95_latency_ms == 150


def test_summarize_tool_calls_handles_empty_input():
    summary = summarize_tool_calls([])

    assert summary.total_calls == 0
    assert summary.failed_calls == 0
    assert summary.success_rate == 0.0
    assert summary.avg_latency_ms == 0.0
    assert summary.p95_latency_ms == 0
    assert summary.tools == []


def test_tool_metrics_requires_authentication(client):
    response = client.get("/api/metrics/tools")
    assert response.status_code == 401


def test_tool_metrics_returns_empty_window(client, new_user):
    response = client.get(
        "/api/metrics/tools",
        params={"tool": "tool-that-does-not-exist"},
        headers=new_user["headers"],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_calls"] == 0
    assert body["tools"] == []
