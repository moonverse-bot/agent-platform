"""重排:词法覆盖度打分、API 重排器的解析、以及按配置选择实现。

重排是「可选的增强」,所以这里既验证它算得对,也验证它在没配置、没 Key 的时候
不会把整条链路带崩 —— 后者和前者一样重要。
"""

import json

import httpx
import pytest

from app.agent import rerank as rerank_module
from app.agent.rerank import (
    LexicalReranker,
    OpenAICompatibleReranker,
    build_reranker,
)
from app.config import Settings


def _settings(**overrides) -> Settings:
    # 不传 key/model 时显式给空串,避免受运行环境的变量影响
    overrides.setdefault("rerank_api_key", "")
    overrides.setdefault("rerank_model", "")
    return Settings(**overrides)


DOCS = [
    "发票申请流程\n在费用中心提交开票申请,需填写公司名称与税号。",
    "退款政策\n自购买之日起 7 天内支持无理由退款,到账时间 3-5 个工作日。",
]


async def test_lexical_rerank_puts_the_document_that_covers_more_query_words_first():
    hits = await LexicalReranker().rerank("退款多久到账", DOCS, top_n=2)

    assert [index for index, _ in hits] == [1, 0]
    best = dict(hits)[1]
    assert best == pytest.approx(2 / 5)  # 5 个二元组里命中了「退款」「到账」
    assert dict(hits)[0] == 0.0


async def test_lexical_rerank_breaks_ties_by_how_often_the_words_appear():
    """覆盖度一样时,同一个词反复出现说明候选更聚焦,应当排在前面。"""
    hits = await LexicalReranker().rerank("退款", ["退款", "退款退款退款"], top_n=2)
    assert [index for index, _ in hits] == [1, 0]
    assert {score for _, score in hits} == {1.0}


async def test_lexical_rerank_scores_stay_within_zero_and_one():
    hits = await LexicalReranker().rerank(
        "退款多久到账 退款多久到账", ["退款到账", "无关内容"], top_n=5
    )
    assert len(hits) == 2
    assert all(0.0 <= score <= 1.0 for _, score in hits)


async def test_lexical_rerank_keeps_the_recall_order_when_there_is_nothing_to_score():
    hits = await LexicalReranker().rerank("!!! ???", ["甲", "乙", "丙"], top_n=2)
    assert hits == [(0, 0.0), (1, 0.0)]


def test_build_reranker_follows_configuration():
    assert build_reranker(_settings(rerank_enabled=False)) is None
    assert isinstance(build_reranker(_settings()), LexicalReranker)
    # 选了 api 却没给 Key:退回零依赖实现,而不是每次调用都报错
    assert isinstance(
        build_reranker(_settings(rerank_provider="api")), LexicalReranker
    )
    assert isinstance(
        build_reranker(_settings(rerank_provider="api", rerank_api_key="sk-x")),
        OpenAICompatibleReranker,
    )


async def test_api_reranker_parses_scores_and_drops_out_of_range_indices(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                    {"index": 9, "relevance_score": 0.99},  # 越界,必须丢掉
                ]
            },
        )

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(rerank_module.httpx, "AsyncClient", factory)

    reranker = OpenAICompatibleReranker(
        api_key="sk-x", base_url="https://example.com/v1", model="bge-reranker"
    )
    hits = await reranker.rerank("退款多久到账", ["文档甲", "文档乙"], top_n=2)

    assert hits == [(1, 0.91), (0, 0.42)]
    assert captured["url"] == "https://example.com/v1/rerank"
    assert captured["body"]["model"] == "bge-reranker"
    assert captured["body"]["documents"] == ["文档甲", "文档乙"]
    assert captured["body"]["top_n"] == 2


async def test_api_reranker_accepts_the_alternative_score_field(monkeypatch):
    """有的服务返回的是 score 而不是 relevance_score,别因为字段名不同就丢掉结果。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "score": 0.7}]})

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(rerank_module.httpx, "AsyncClient", factory)

    reranker = OpenAICompatibleReranker(
        api_key="sk-x", base_url="https://example.com/v1", model="m"
    )
    assert await reranker.rerank("q", ["文档"], top_n=1) == [(0, 0.7)]
    assert await reranker.rerank("q", [], top_n=1) == []
