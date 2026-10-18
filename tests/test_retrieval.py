"""检索层:向量化、向量存储、RRF 融合与降级。

测试环境没有配 Embedding Key,所以走的是离线哈希向量。它一样会经过
「向量化 → 落库 → 相似度检索 → 排名融合」这条完整链路,只是召回质量取决于
词形重合,不能拿来做效果评估 —— 真正的效果要用神经 Embedding 才看得到。

Postgres 那条路径本机跑不了(没有 Docker),所以退一步断言生成的 SQL:
距离算子、类型转换、HNSW 索引这些少一个都会在线上静默变慢或报错。
"""

import json

import httpx
import pytest
from sqlalchemy import delete, select

from app.agent import embeddings
from app.agent.embeddings import (
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
    clear_embedding_cache,
    cosine_similarity,
    embed_cached,
    l2_normalize,
)
from app.agent.retrieval import (
    ChunkHit,
    ChunkVectorRecord,
    PgChunkVectorStore,
    PgVectorStore,
    SqliteChunkVectorStore,
    SqliteVectorStore,
    VectorRecord,
    build_chunk_vector_store,
    build_vector_store,
    chunk_hits_to_article_ranking,
    fuse_rankings,
    merge_vector_rankings,
    reciprocal_rank_fusion,
)
from app.agent.tools import build_default_registry
from app.agent.tools import knowledge as knowledge_tool
from app.agent.tools.base import ToolContext
from app.config import Settings
from app.db.init_db import SEED_ARTICLES as SEED_ARTICLE_ROWS
from app.db.init_db import refresh_knowledge_chunks, refresh_knowledge_index
from app.db.models import KnowledgeArticle, KnowledgeChunk, KnowledgeVector

# 断言索引覆盖率时以种子数据为准,不写死条数 —— 知识库以后还会继续变长
SEED_ARTICLES = len(SEED_ARTICLE_ROWS)


def _settings(**overrides) -> Settings:
    # 不传 embedding_api_key 时显式给空串,避免受运行环境的变量影响
    overrides.setdefault("embedding_api_key", "")
    overrides.setdefault("embedding_model", "")
    return Settings(**overrides)


def _mock_http(monkeypatch, handler) -> None:
    """把 embeddings 模块用到的 httpx.AsyncClient 换成 MockTransport,测试里不发真实请求。

    必须先取出原始类再替换:httpx 是同一个模块对象,直接把工厂函数塞进去,
    工厂内部就会调到它自己,变成无限递归。
    """
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", factory)


@pytest.fixture(autouse=True)
def _reset_vector_circuit():
    """向量熔断状态是模块级全局的,任其残留会把别的用例一起带崩。"""
    knowledge_tool.reset_vector_circuit()
    yield
    knowledge_tool.reset_vector_circuit()


@pytest.fixture
async def make_article(db_user: ToolContext):
    """造一篇只属于本用例的知识库文章,用完连同它的向量一起删掉。

    测试库是整个会话共享的,直接往种子文章上写向量会污染其它用例的索引断言。
    """
    created: list[KnowledgeArticle] = []

    async def _make(title: str, content: str, tags: str = "测试") -> KnowledgeArticle:
        article = KnowledgeArticle(title=title, content=content, tags=tags)
        db_user.session.add(article)
        await db_user.session.commit()
        await db_user.session.refresh(article)
        created.append(article)
        return article

    yield _make

    for article in created:
        await db_user.session.execute(
            delete(KnowledgeVector).where(KnowledgeVector.article_id == article.id)
        )
        # 段落行也要显式删掉:级联删除依赖外键开关,测试库不保证开着,
        # 留下孤儿段落会让后面的「索引覆盖率」断言莫名其妙地失败
        await db_user.session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.article_id == article.id)
        )
        await db_user.session.delete(article)
    await db_user.session.commit()


# ---------------- 向量化 ----------------


def test_l2_normalize_produces_unit_vector():
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_similarity_ignores_vector_length():
    assert cosine_similarity([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 2.0]) == pytest.approx(0.0)
    # 维度对不上时返回 0,而不是抛异常把整条检索链路打断
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


async def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=256)
    left, right = await embedder.embed(["退款政策", "退款政策"])

    assert left == right
    assert len(left) == 256
    assert sum(value * value for value in left) == pytest.approx(1.0)
    assert await embedder.embed([""]) == [[0.0] * 256]


async def test_hashing_embedder_scores_related_text_higher():
    embedder = HashingEmbedder(dim=1024)
    query, related, unrelated = await embedder.embed(
        [
            "退款要多久到账",
            "退款政策 自购买之日起 7 天内支持无理由退款",
            "如何导出历史数据 CSV 报表",
        ]
    )
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_build_embedder_prefers_online_but_degrades_to_local():
    assert isinstance(build_embedder(_settings(embedding_provider="auto")), HashingEmbedder)

    online = build_embedder(
        _settings(
            embedding_provider="auto", embedding_api_key="sk-test", embedding_model="bge-m3"
        )
    )
    assert isinstance(online, OpenAICompatibleEmbedder)
    assert online.name == "openai-compatible:bge-m3"

    # 只写了 provider 却忘了给 Key,应当告警降级,而不是等到请求时报错
    assert isinstance(
        build_embedder(_settings(embedding_provider="openai", embedding_model="bge-m3")),
        HashingEmbedder,
    )


async def test_embedding_cache_avoids_second_round_trip():
    clear_embedding_cache()
    calls: list[list[str]] = []

    class CountingEmbedder:
        name = "counting-embedder"
        dim = 2
        default_min_score = 0.0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return [[float(len(text)), 1.0] for text in texts]

    embedder = CountingEmbedder()
    first = await embed_cached(embedder, ["退款"])
    second = await embed_cached(embedder, ["退款", "退款"])

    assert calls == [["退款"]]
    assert first[0] == second[0] == second[1]
    clear_embedding_cache()


async def test_openai_embedder_respects_response_order(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        # 故意把 index 打乱,验证我们按 index 还原顺序
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    _mock_http(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        api_key="sk-test", base_url="https://example.com/v1/", model="bge-m3", dim=2
    )

    vectors = await embedder.embed(["甲", "乙"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://example.com/v1/embeddings"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"] == {"model": "bge-m3", "input": ["甲", "乙"]}


async def test_openai_embedder_rejects_dimension_drift(monkeypatch):
    """模型维度与配置不一致时必须立刻报错,否则索引里会混进不同维度的向量。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
        )

    _mock_http(monkeypatch, handler)
    embedder = OpenAICompatibleEmbedder(
        api_key="sk-test", base_url="https://example.com/v1", model="bge-m3", dim=1024
    )

    with pytest.raises(ValueError, match="维度不匹配"):
        await embedder.embed(["x"])


# ---------------- 向量存储 ----------------


async def test_sqlite_store_searches_in_cosine_order(db_user: ToolContext, make_article):
    article = await make_article("临时文档", "仅供向量存储测试使用")
    store = SqliteVectorStore()
    await store.ensure_schema(db_user.session)
    await store.upsert(
        db_user.session,
        [
            VectorRecord(
                article_id=article.id, embedder="unit-test", dim=2, vector=[1.0, 0.0]
            )
        ],
    )

    hits = await store.search(
        db_user.session, vector=[1.0, 0.0], embedder="unit-test", dim=2, limit=5
    )
    assert hits[0][0] == article.id
    assert hits[0][1] == pytest.approx(1.0)

    # 同一篇文章再写一次是覆盖而不是新增,否则重复索引会越滚越大
    await store.upsert(
        db_user.session,
        [
            VectorRecord(
                article_id=article.id, embedder="unit-test", dim=2, vector=[0.0, 1.0]
            )
        ],
    )
    assert await store.indexed_ids(db_user.session, embedder="unit-test", dim=2) == {
        article.id
    }
    assert await store.search(
        db_user.session, vector=[1.0, 0.0], embedder="unit-test", dim=2, limit=5
    ) == [(article.id, 0.0)]


async def test_vector_rows_are_scoped_to_embedder_and_dim(db_user: ToolContext, make_article):
    """换了 Embedding 模型或维度,旧向量必须被当成「没有」,否则会拿错版本算相似度。"""
    article = await make_article("临时文档", "仅供向量存储测试使用")
    store = SqliteVectorStore()
    await store.upsert(
        db_user.session,
        [
            VectorRecord(
                article_id=article.id, embedder="model-a", dim=2, vector=[1.0, 0.0]
            )
        ],
    )

    assert await store.indexed_ids(db_user.session, embedder="model-b", dim=2) == set()
    assert await store.indexed_ids(db_user.session, embedder="model-a", dim=3) == set()
    assert await store.indexed_ids(db_user.session, embedder="model-a", dim=2) == {
        article.id
    }
    assert await store.search(
        db_user.session, vector=[1.0, 0.0], embedder="model-a", dim=3, limit=5
    ) == []


def test_build_vector_store_selects_backend():
    portable = build_vector_store(_settings(database_url="sqlite+aiosqlite:///./x.db"))
    assert isinstance(portable, SqliteVectorStore)

    pg_settings = _settings(database_url="postgresql+asyncpg://agent:agent@localhost:5432/agent")
    pg_store = build_vector_store(pg_settings)
    assert isinstance(pg_store, PgVectorStore)
    assert pg_store.dim == pg_settings.embedding_dim

    assert build_vector_store(_settings(vector_backend="none")) is None
    # 后端与数据库方言不匹配时静默关掉,而不是等到运行时崩在 SQL 上
    assert (
        build_vector_store(
            _settings(database_url="sqlite+aiosqlite:///./x.db", vector_backend="pgvector")
        )
        is None
    )
    # 也可以强制用便携后端
    assert isinstance(
        build_vector_store(
            _settings(
                database_url="postgresql+asyncpg://agent:agent@localhost:5432/agent",
                vector_backend="sqlite",
            )
        ),
        SqliteVectorStore,
    )


def test_pgvector_schema_and_queries_use_native_vector_features():
    store = PgVectorStore(dim=1024)

    schema = "\n".join(store.schema_statements())
    assert "CREATE EXTENSION IF NOT EXISTS vector" in schema
    assert "vector(1024)" in schema
    assert "USING hnsw (embedding vector_cosine_ops)" in schema
    assert "ON DELETE CASCADE" in schema

    search = store.search_sql()
    assert "embedding <=> CAST(:embedding AS vector)" in search
    assert "ORDER BY embedding <=> CAST(:embedding AS vector)" in search
    assert "WHERE embedder = :embedder AND dim = :dim" in search
    # 余弦距离换回相似度,分数方向要和关键词一路一致(越大越相关)
    assert "1 - (embedding <=>" in search

    upsert = store.upsert_sql()
    assert "ON CONFLICT (article_id) DO UPDATE" in upsert
    assert "CAST(:embedding AS vector)" in upsert

    assert store.indexed_ids_sql().startswith("SELECT article_id FROM knowledge_embeddings")


def test_pgvector_literal_is_a_plain_number_list():
    literal = PgVectorStore._literal([0.5, -1.25])
    assert literal == "[0.50000000,-1.25000000]"
    # 向量是拼进 SQL 文本的,所以必须保证只可能出现数字与括号,没有注入面
    assert set(literal) <= set("[]0123456789.,-")


# ---------------- 融合排序与降级 ----------------


def test_reciprocal_rank_fusion_rewards_articles_hit_by_both_paths():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4]])
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    # 两路都命中的 3 一定排在只被关键词命中的 1 前面
    assert fused[3] > fused[1]
    assert 4 in fused and 2 in fused


def test_fuse_rankings_keeps_both_ranks_for_explanation():
    fused = fuse_rankings([5, 6], [(6, 0.9), (7, 0.8)])

    assert [item.article_id for item in fused][0] == 6
    assert fused[0].keyword_rank == 2
    assert fused[0].vector_rank == 1
    assert fused[0].vector_score == pytest.approx(0.9)

    by_id = {item.article_id: item for item in fused}
    assert by_id[7].keyword_rank is None
    assert by_id[5].vector_rank is None


async def test_refresh_knowledge_index_covers_every_article(db_user: ToolContext):
    # 第一次可能是 init_db 顺手建好的,这里直接 force 重算,保证断言与执行顺序无关
    assert await refresh_knowledge_index(force=True) == SEED_ARTICLES
    assert await refresh_knowledge_index() == 0

    indexed = await SqliteVectorStore().indexed_ids(
        db_user.session, embedder=HashingEmbedder.name, dim=1024
    )
    articles = list(await db_user.session.scalars(select(KnowledgeArticle.id)))
    assert set(articles) <= indexed


async def test_hybrid_mode_combines_keyword_and_vector(db_user: ToolContext, monkeypatch):
    await refresh_knowledge_index()
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="hybrid")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "退款要多久到账"}, db_user
    )
    payload = json.loads(result.output)

    assert payload["retrieval_mode"] == "hybrid"
    assert "sqlite-json:local-hash-bigram" in payload["vector"]
    best = payload["results"][0]
    assert best["title"] == "退款政策"
    # 两路都命中,所以两个排名都要有,前端/日志才解释得清它为什么排第一
    assert best["keyword_rank"] == 1
    assert best["vector_rank"] == 1


async def test_vector_only_mode_still_finds_the_article(db_user: ToolContext, monkeypatch):
    await refresh_knowledge_index()
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="vector")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "退款要多久到账"}, db_user
    )
    payload = json.loads(result.output)

    best = payload["results"][0]
    assert best["title"] == "退款政策"
    assert best["keyword_rank"] is None  # 说明这一路确实没走关键词
    assert best["vector_rank"] == 1


async def test_vector_unavailable_degrades_to_keyword(db_user: ToolContext, monkeypatch):
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(vector_backend="none")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "退款要多久到账"}, db_user
    )
    payload = json.loads(result.output)

    assert payload["vector"] == "vector-backend=none"
    assert payload["results"][0]["title"] == "退款政策"
    assert payload["results"][0]["vector_rank"] is None


async def test_embedding_failure_degrades_to_keyword(db_user: ToolContext, monkeypatch):
    """向量服务抖动不能让检索工具直接失败 —— 关键词一路必须顶上。"""

    class BrokenEmbedder:
        name = "broken"
        dim = 1024
        default_min_score = 0.0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding 服务 502")

    monkeypatch.setattr(knowledge_tool, "build_embedder", lambda settings: BrokenEmbedder())
    monkeypatch.setattr(knowledge_tool, "get_settings", lambda: _settings())

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "退款要多久到账"}, db_user
    )
    payload = json.loads(result.output)

    assert result.ok is True
    assert payload["vector"] == "embedding-failed"
    assert payload["results"][0]["title"] == "退款政策"


async def test_noise_is_filtered_by_the_similarity_threshold(
    db_user: ToolContext, monkeypatch
):
    """阈值必须挡得住「语义上无关但余弦不为 0」的噪声,否则会把不相关文章喂给模型。"""
    await refresh_knowledge_index()
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="vector")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "量子纠缠对股票市场的影响"}, db_user
    )
    assert "没有找到" in result.output


async def test_vector_channel_surfaces_what_keyword_misses(
    db_user: ToolContext, make_article, monkeypatch
):
    """向量这一路的价值就是「换个说法也能召回」:关键词一个字都没命中时,向量能把它捞回来。

    离线哈希向量做不到这件事(它只能匹配词形重合),所以这里注入一个懂同义词的假 embedder,
    验证的是融合机制本身 —— 只要向量给出真实信号,关键词漏掉的文档就会排到最前面。
    """
    hidden = await make_article(
        "售后资金处理说明", "符合条件的订单可申请资金处理,3-5 个工作日到账。"
    )

    class SynonymEmbedder:
        """口语的「钱 / 回来」和书面的「资金」落到同一个方向上。"""

        name = "stub-synonym"
        dim = 8
        default_min_score = 0.5

        async def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = []
            for text in texts:
                vector = [0.0] * 8
                if "资金" in text or "回来" in text:
                    vector[0] = 1.0
                vectors.append(vector)
            return vectors

    embedder = SynonymEmbedder()
    await SqliteVectorStore().upsert(
        db_user.session,
        [
            VectorRecord(
                article_id=hidden.id,
                embedder=embedder.name,
                dim=embedder.dim,
                vector=(await embedder.embed([hidden.title + hidden.content]))[0],
            )
        ],
    )

    monkeypatch.setattr(knowledge_tool, "build_embedder", lambda settings: embedder)
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="hybrid")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "钱啥时候能回来"}, db_user
    )
    payload = json.loads(result.output)

    assert [item["title"] for item in payload["results"]] == ["售后资金处理说明"]
    best = payload["results"][0]
    assert best["keyword_rank"] is None  # 关键词一路确实一个字都没命中
    assert best["vector_rank"] == 1
    assert best["vector_score"] == pytest.approx(1.0)


async def test_repeated_embedding_failures_stop_retrying(
    db_user: ToolContext, monkeypatch
):
    """Embedding 服务挂了之后,冷却期内不该再为每次提问都付一次超时。"""
    attempts: list[list[str]] = []

    class BrokenEmbedder:
        name = "broken-circuit"
        dim = 1024
        default_min_score = 0.0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            attempts.append(list(texts))
            raise RuntimeError("embedding 服务 502")

    monkeypatch.setattr(knowledge_tool, "build_embedder", lambda settings: BrokenEmbedder())
    monkeypatch.setattr(knowledge_tool, "get_settings", lambda: _settings())
    registry = build_default_registry()

    first = json.loads(
        (await registry.execute("search_knowledge", {"query": "退款要多久到账"}, db_user)).output
    )
    second = json.loads(
        (await registry.execute("search_knowledge", {"query": "发票"}, db_user)).output
    )

    assert first["vector"] == "embedding-failed"
    assert second["vector"] == "vector-circuit-open"
    assert len(attempts) == 1  # 只真的试过一次
    # 冷却期内关键词一路照样把问题答了
    assert second["results"][0]["title"] == "发票申请流程"


# ---------------- 段落级索引与重排 ----------------


async def test_chunk_index_is_built_for_every_article_and_is_idempotent(
    db_user: ToolContext,
):
    assert await refresh_knowledge_chunks(force=True) > 0
    assert await refresh_knowledge_chunks() == 0

    rows = list(await db_user.session.scalars(select(KnowledgeChunk)))
    articles = list(await db_user.session.scalars(select(KnowledgeArticle)))
    by_article: dict[int, list[KnowledgeChunk]] = {}
    for row in rows:
        by_article.setdefault(row.article_id, []).append(row)

    assert set(by_article) == {article.id for article in articles}
    # 每块都得有当前模型的向量,否则检索时会静默漏掉它
    assert all(row.vector and row.embedder for row in rows)
    # 切块的意义就在于长文档会被切成多块,短文档仍然是一块
    assert max(len(chunks) for chunks in by_article.values()) > 1


def test_pgvector_chunk_schema_partitions_the_index_by_embedder():
    """pgvector 那条路径本机跑不了(没有 Docker),退一步断言生成的 SQL 形状。"""
    store = PgChunkVectorStore(dim=1024, embedder="openai-compatible:BAAI/bge-m3")
    ddl = "\n".join(store.schema_statements())
    suffix = "openai_compatible_baai_bge_m3"

    assert "vector(1024)" in ddl
    # 不同 Embedding 模型的向量空间不可比,唯一索引与 HNSW 都必须按 embedder 分区
    assert f"knowledge_chunk_embeddings_{suffix}_key_idx" in ddl
    assert f"knowledge_chunk_embeddings_{suffix}_hnsw_idx" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert ddl.count("WHERE embedder = 'openai-compatible:BAAI/bge-m3'") == 2

    search = store.search_sql()
    # 命中的段落要连正文一起取回来,否则还得再查一次数据库
    assert "JOIN knowledge_chunks c" in search
    assert "e.embedding <=> CAST(:embedding AS vector)" in search
    assert "WHERE e.embedder = :embedder AND e.dim = :dim" in search

    upsert = store.upsert_sql()
    assert "ON CONFLICT (article_id, ordinal)" in upsert
    assert "WHERE embedder = 'openai-compatible:BAAI/bge-m3'" in upsert
    assert store.drop_sql() == (
        "DELETE FROM knowledge_chunk_embeddings WHERE article_id = :article_id"
    )


def test_build_chunk_vector_store_follows_the_vector_backend_setting():
    assert isinstance(
        build_chunk_vector_store(_settings(), embedder="local"), SqliteChunkVectorStore
    )
    assert build_chunk_vector_store(_settings(vector_backend="none"), embedder="local") is None
    # 声明用 pgvector 但数据库不是 Postgres:关掉段落检索,而不是启动时报错
    assert (
        build_chunk_vector_store(
            _settings(vector_backend="pgvector", database_url="sqlite+aiosqlite:///./x.db"),
            embedder="local",
        )
        is None
    )


def test_chunk_hits_collapse_to_the_best_chunk_per_article():
    """一篇文档命中多块时只算一次,取最好的那块 —— 求和会让长文档天然占优。"""
    hits = [
        ChunkHit(article_id=1, ordinal=0, score=0.2, content="弱命中"),
        ChunkHit(article_id=1, ordinal=1, score=0.7, content="强命中"),
        ChunkHit(article_id=2, ordinal=0, score=0.5, content="另一篇"),
    ]
    assert chunk_hits_to_article_ranking(hits) == [(1, 0.7), (2, 0.5)]


def test_merge_vector_rankings_uses_rrf_and_reports_the_better_score():
    merged = merge_vector_rankings([(1, 0.9), (2, 0.1)], [(2, 0.95), (1, 0.05)])
    assert [article_id for article_id, _ in merged] == [1, 2]
    # 展示用的分数取两路里更高的那个,而不是 RRF 的融合分
    assert dict(merged) == {1: 0.9, 2: 0.95}


def test_merge_vector_rankings_passes_through_a_single_path():
    assert merge_vector_rankings([(1, 0.9)], []) == [(1, 0.9)]
    assert merge_vector_rankings([], [(2, 0.5)]) == [(2, 0.5)]
    assert merge_vector_rankings([], []) == []


async def test_chunk_recall_returns_the_matching_paragraph(
    db_user: ToolContext, make_article, monkeypatch
):
    """段落级召回的价值:一篇长文档里只有一小段讲这件事时,能把那一小段原样捞出来。

    离线哈希向量做不到「换个说法也能命中」,所以这里注入一个懂主题的假 embedder,
    验证的是「命中段落 → 折算成文章 → 把该段作为 matched_chunk 返回」这条链路。
    """
    article = await make_article("夜间服务说明", "白天与夜间的服务安排见下面的分段说明。")

    class TopicEmbedder:
        name = "stub-topic"
        dim = 8
        default_min_score = 0.5

        async def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = []
            for text in texts:
                vector = [0.0] * 8
                if "夜间" in text:
                    vector[0] = 1.0
                if "白天" in text:
                    vector[1] = 1.0
                vectors.append(vector)
            return vectors

    embedder = TopicEmbedder()
    chunks = [
        "夜间服务说明\n夜间只保留智能助手自动应答,人工客服次日上午处理。",
        "夜间服务说明\n白天人工客服 9:00 - 18:00 在线,可以实时对话。",
    ]
    db_user.session.add_all(
        KnowledgeChunk(article_id=article.id, ordinal=index, content=text)
        for index, text in enumerate(chunks)
    )
    await db_user.session.commit()
    await SqliteChunkVectorStore().upsert_chunks(
        db_user.session,
        [
            ChunkVectorRecord(
                article_id=article.id,
                ordinal=index,
                embedder=embedder.name,
                dim=embedder.dim,
                vector=(await embedder.embed([text]))[0],
            )
            for index, text in enumerate(chunks)
        ],
    )

    monkeypatch.setattr(knowledge_tool, "build_embedder", lambda settings: embedder)
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="vector")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "夜间没人值班的时候怎么办"}, db_user
    )
    payload = json.loads(result.output)

    assert [item["title"] for item in payload["results"]] == ["夜间服务说明"]
    best = payload["results"][0]
    assert best["vector_rank"] == 1
    assert best["vector_score"] == pytest.approx(1.0)
    assert "夜间只保留智能助手" in best["matched_chunk"]
    assert "白天人工客服" not in best["matched_chunk"]


async def test_rerank_scores_are_reported_with_the_reranker_name(
    db_user: ToolContext, monkeypatch
):
    await refresh_knowledge_index()
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="hybrid")
    )

    payload = json.loads(
        (
            await build_default_registry().execute(
                "search_knowledge", {"query": "退款要多久到账"}, db_user
            )
        ).output
    )

    assert payload["reranker"] == "lexical-coverage"
    best = payload["results"][0]
    assert best["title"] == "退款政策"
    assert 0 < best["rerank_score"] <= 1


async def test_rerank_failure_falls_back_to_the_recall_order(
    db_user: ToolContext, monkeypatch
):
    """重排是可选增强:它挂了顶多回到召回顺序,绝不能让检索失败。"""

    class BrokenReranker:
        name = "broken"

        async def rerank(self, query, documents, *, top_n):
            raise RuntimeError("重排服务 503")

    await refresh_knowledge_index()
    monkeypatch.setattr(
        knowledge_tool, "build_reranker", lambda settings: BrokenReranker()
    )
    monkeypatch.setattr(
        knowledge_tool, "get_settings", lambda: _settings(retrieval_mode="hybrid")
    )

    result = await build_default_registry().execute(
        "search_knowledge", {"query": "退款要多久到账"}, db_user
    )
    payload = json.loads(result.output)

    assert result.ok is True
    assert payload["reranker"] is None
    assert payload["results"][0]["title"] == "退款政策"
    assert payload["results"][0]["rerank_score"] is None
