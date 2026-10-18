"""健康检查与链路追踪。"""


def test_healthz_exposes_runtime_state(client):
    response = client.get("/healthz")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    # 测试环境没有配 API Key,应当如实报告 Agent 不可用
    assert body["llm_enabled"] is False
    # 没配 Redis 时应当退化到进程内限流
    assert body["rate_limit_backend"] == "memory"
    assert "search_knowledge" in body["tools"]

    # 检索配置要能一眼看到:SQLite 走便携后端,没配 Embedding 时用离线兜底
    assert body["retrieval_mode"] == "hybrid"
    assert body["vector_backend"] == "sqlite-json"
    assert body["embedder"] == "local-hash-bigram-v2"

    # 段落级索引与重排也要看得见:切块没生效、重排被关掉,一眼就能发现
    assert body["chunk_backend"] == "sqlite-json"
    assert body["chunk_size"] == 300
    assert body["chunks_indexed"] > 0
    assert body["reranker"] == "lexical-coverage"


def test_trace_id_is_echoed_back(client):
    response = client.get("/healthz", headers={"X-Trace-Id": "trace-abc"})
    assert response.headers["X-Trace-Id"] == "trace-abc"


def test_trace_id_is_generated_when_absent(client):
    response = client.get("/healthz")
    assert len(response.headers["X-Trace-Id"]) == 16
