"""知识库召回:向量存储与多路排名融合。

向量存储做成同一套接口的两个实现,靠配置切换:

  * `PgVectorStore`      —— Postgres + pgvector 扩展,列类型是原生 `vector(N)`,
                            距离算子 `<=>` 走 HNSW 索引,检索在数据库里完成;
  * `SqliteVectorStore`  —— SQLite(本地开发与测试),向量以 JSON 文本落库,
                            余弦在进程内算。数据量大时这是全表扫描,所以生产用 pgvector。

同一套接口在「段落级」上再实现一遍(`SqliteChunkVectorStore` / `PgChunkVectorStore`):
整篇一个向量和每段一个向量回答的是不同粒度的问题,所以两路都留着,
在 `merge_vector_rankings` 里合成一路「向量召回」。

两路召回(关键词、向量)用 RRF(Reciprocal Rank Fusion)融合:它只看排名不看分数,
天然绕开「关键词分是整数、余弦是 0~1 小数」这种量纲不可比的问题,
也不需要为两路分数调权重。
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.embeddings import cosine_similarity
from app.config import Settings
from app.db.models import KnowledgeChunk, KnowledgeVector

logger = logging.getLogger(__name__)

# RRF 的平滑常数:排名越靠后分越低,k 越大则排名差异被压得越平。
# 60 是原论文给的默认值。
RRF_K = 60


@dataclass
class VectorRecord:
    article_id: int
    embedder: str
    dim: int
    vector: list[float]


@dataclass
class ChunkVectorRecord:
    """一段文档的向量。ordinal 是它在这篇文档里的序号,回填正文时要靠它。"""

    article_id: int
    ordinal: int
    embedder: str
    dim: int
    vector: list[float]


@dataclass
class ChunkHit:
    """段落级检索的一条命中。content 为「命中的那一段原文」,直接给模型看。"""

    article_id: int
    ordinal: int
    score: float
    content: str = ""


@dataclass
class RankedArticle:
    """融合后的单条结果,保留两路各自的排名,方便在工具输出里解释「为什么召回它」。"""

    article_id: int
    fusion_score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    vector_score: float | None = None


class VectorStore(Protocol):
    name: str

    async def ensure_schema(self, session: AsyncSession) -> None: ...

    async def upsert(self, session: AsyncSession, records: Sequence[VectorRecord]) -> int: ...

    async def search(
        self,
        session: AsyncSession,
        *,
        vector: list[float],
        embedder: str,
        dim: int,
        limit: int,
    ) -> list[tuple[int, float]]: ...

    async def indexed_ids(
        self, session: AsyncSession, *, embedder: str, dim: int
    ) -> set[int]: ...


class ChunkVectorStore(Protocol):
    """段落级向量存储。方法与文章级一一对应,只是多了一个 ordinal。"""

    name: str

    async def ensure_schema(self, session: AsyncSession) -> None: ...

    async def upsert_chunks(
        self, session: AsyncSession, records: Sequence[ChunkVectorRecord]
    ) -> int: ...

    async def search_chunks(
        self,
        session: AsyncSession,
        *,
        vector: list[float],
        embedder: str,
        dim: int,
        limit: int,
    ) -> list[ChunkHit]: ...

    async def drop_article_chunks(
        self, session: AsyncSession, *, article_id: int
    ) -> None:
        """丢掉一篇文章的全部段落向量。正文变更后重新切块时调用,避免残留旧块。"""
        ...


class SqliteVectorStore:
    """SQLite 实现:向量存 JSON 文本,进程内算余弦。"""

    name = "sqlite-json"

    async def ensure_schema(self, session: AsyncSession) -> None:
        # 表由 Base.metadata.create_all 建好,这里不用额外处理
        return None

    async def upsert(self, session: AsyncSession, records: Sequence[VectorRecord]) -> int:
        for record in records:
            payload = json.dumps(record.vector)
            existing = await session.get(KnowledgeVector, record.article_id)
            if existing is None:
                session.add(
                    KnowledgeVector(
                        article_id=record.article_id,
                        embedder=record.embedder,
                        dim=record.dim,
                        vector=payload,
                    )
                )
            else:
                existing.embedder = record.embedder
                existing.dim = record.dim
                existing.vector = payload
        await session.commit()
        return len(records)

    async def search(
        self,
        session: AsyncSession,
        *,
        vector: list[float],
        embedder: str,
        dim: int,
        limit: int,
    ) -> list[tuple[int, float]]:
        rows = await session.scalars(
            select(KnowledgeVector).where(
                KnowledgeVector.embedder == embedder, KnowledgeVector.dim == dim
            )
        )
        scored = [
            (row.article_id, cosine_similarity(vector, json.loads(row.vector))) for row in rows
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    async def indexed_ids(
        self, session: AsyncSession, *, embedder: str, dim: int
    ) -> set[int]:
        rows = await session.scalars(
            select(KnowledgeVector.article_id).where(
                KnowledgeVector.embedder == embedder, KnowledgeVector.dim == dim
            )
        )
        return set(rows)


class PgVectorStore:
    """Postgres + pgvector 实现。

    刻意不走 pgvector 的 Python 包:向量以 `[0.1,0.2,...]` 文本参数传入,
    在 SQL 里用 `CAST(:embedding AS vector)` 转类型,这样 SQLite 用户不需要装任何新依赖。
    """

    name = "pgvector"
    table = "knowledge_embeddings"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def schema_statements(self) -> list[str]:
        """建扩展 / 建表 / 建索引的 DDL。抽成方法是为了在没有 Postgres 的环境里也能断言它。"""
        return [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    article_id integer PRIMARY KEY
                        REFERENCES knowledge_articles(id) ON DELETE CASCADE,
                    embedder varchar(64) NOT NULL,
                    dim integer NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """,
            # HNSW + 余弦距离:建好之后检索是近似最近邻,不用全表扫
            f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
            f"ON {self.table} USING hnsw (embedding vector_cosine_ops)",
        ]

    def upsert_sql(self) -> str:
        return f"""
            INSERT INTO {self.table} (article_id, embedder, dim, embedding)
            VALUES (:article_id, :embedder, :dim, CAST(:embedding AS vector))
            ON CONFLICT (article_id) DO UPDATE SET
                embedder = EXCLUDED.embedder,
                dim = EXCLUDED.dim,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """

    def search_sql(self) -> str:
        return f"""
            SELECT article_id, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM {self.table}
            WHERE embedder = :embedder AND dim = :dim
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """

    def indexed_ids_sql(self) -> str:
        return (
            f"SELECT article_id FROM {self.table} "
            "WHERE embedder = :embedder AND dim = :dim"
        )

    async def ensure_schema(self, session: AsyncSession) -> None:
        for statement in self.schema_statements():
            await session.execute(text(statement))
        await session.commit()

    @staticmethod
    def _literal(vector: Sequence[float]) -> str:
        # 只由浮点数拼成,不存在注入面;用文本传参是为了不引入 pgvector 的 Python 包
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

    async def upsert(self, session: AsyncSession, records: Sequence[VectorRecord]) -> int:
        statement = text(self.upsert_sql())
        for record in records:
            await session.execute(
                statement,
                {
                    "article_id": record.article_id,
                    "embedder": record.embedder,
                    "dim": record.dim,
                    "embedding": self._literal(record.vector),
                },
            )
        await session.commit()
        return len(records)

    async def search(
        self,
        session: AsyncSession,
        *,
        vector: list[float],
        embedder: str,
        dim: int,
        limit: int,
    ) -> list[tuple[int, float]]:
        statement = text(self.search_sql())
        rows = await session.execute(
            statement,
            {
                "embedding": self._literal(vector),
                "embedder": embedder,
                "dim": dim,
                "limit": limit,
            },
        )
        return [(row[0], float(row[1])) for row in rows]

    async def indexed_ids(
        self, session: AsyncSession, *, embedder: str, dim: int
    ) -> set[int]:
        rows = await session.execute(
            text(self.indexed_ids_sql()), {"embedder": embedder, "dim": dim}
        )
        return {row[0] for row in rows}


class SqliteChunkVectorStore:
    """SQLite 实现:段落向量以 JSON 文本落在 knowledge_chunks 表的向量列上。

    正文和向量同一行,所以「删正文」就等于「删向量」,不需要额外的清理逻辑。
    """

    name = "sqlite-json"

    async def ensure_schema(self, session: AsyncSession) -> None:
        # 表由 Base.metadata.create_all 建好,这里不用额外处理
        return None

    async def upsert_chunks(
        self, session: AsyncSession, records: Sequence[ChunkVectorRecord]
    ) -> int:
        written = 0
        for record in records:
            row = await session.scalar(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.article_id == record.article_id,
                    KnowledgeChunk.ordinal == record.ordinal,
                )
            )
            if row is None:
                # 正文一定先由 init_db 写进去。这里找不到行说明调用顺序错了,
                # 与其写出一条没有正文的脏数据,不如跳过并告警。
                logger.warning(
                    "段落向量找不到对应正文,已跳过",
                    extra={"event": "retrieval.chunk_missing"},
                )
                continue
            row.embedder = record.embedder
            row.dim = record.dim
            row.vector = json.dumps(record.vector)
            written += 1
        await session.commit()
        return written

    async def search_chunks(
        self,
        session: AsyncSession,
        *,
        vector: list[float],
        embedder: str,
        dim: int,
        limit: int,
    ) -> list[ChunkHit]:
        rows = await session.scalars(
            select(KnowledgeChunk).where(
                KnowledgeChunk.embedder == embedder, KnowledgeChunk.dim == dim
            )
        )
        scored = [
            ChunkHit(
                article_id=row.article_id,
                ordinal=row.ordinal,
                score=cosine_similarity(vector, json.loads(row.vector)),
                content=row.content,
            )
            for row in rows
            if row.vector
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    async def drop_article_chunks(self, session: AsyncSession, *, article_id: int) -> None:
        # 正文与向量是同一行,调用方删正文时就把向量一起删掉了,这里无需额外处理。
        return None


class PgChunkVectorStore:
    """Postgres + pgvector 实现。正文仍从 knowledge_chunks 读,这张表只存向量。

    索引按 embedder 分区:HNSW 只支持单列,而不同 Embedding 模型的向量空间互不可比,
    混在同一个索引里会互相干扰,所以唯一索引与 HNSW 都带上 `WHERE embedder = ...`。
    """

    name = "pgvector"
    table = "knowledge_chunk_embeddings"

    def __init__(self, dim: int, embedder: str) -> None:
        self.dim = dim
        self.embedder = embedder

    def _suffix(self) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", self.embedder.lower()).strip("_")
        return cleaned or "default"

    def _embedder_literal(self) -> str:
        # 只用于拼接 DDL 里带条件的索引;名字来自配置或模型名,这里顺手转义单引号
        return self.embedder.replace("'", "''")

    def schema_statements(self) -> list[str]:
        suffix = self._suffix()
        embedder = self._embedder_literal()
        return [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    article_id integer NOT NULL
                        REFERENCES knowledge_articles(id) ON DELETE CASCADE,
                    ordinal integer NOT NULL,
                    embedder varchar(64) NOT NULL,
                    dim integer NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """,
            # 同一个 (article_id, ordinal) 在每个 embedder 下只允许一条
            f"CREATE UNIQUE INDEX IF NOT EXISTS {self.table}_{suffix}_key_idx "
            f"ON {self.table} (article_id, ordinal) WHERE embedder = '{embedder}'",
            # 只给当前 embedder 建 HNSW:跨模型的向量空间不可比,混在一起索引会失真
            f"CREATE INDEX IF NOT EXISTS {self.table}_{suffix}_hnsw_idx "
            f"ON {self.table} USING hnsw (embedding vector_cosine_ops) "
            f"WHERE embedder = '{embedder}'",
        ]

    def upsert_sql(self) -> str:
        return f"""
            INSERT INTO {self.table} (article_id, ordinal, embedder, dim, embedding)
            VALUES (:article_id, :ordinal, :embedder, :dim, CAST(:embedding AS vector))
            ON CONFLICT (article_id, ordinal) WHERE embedder = '{self._embedder_literal()}'
            DO UPDATE SET
                dim = EXCLUDED.dim,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """

    def search_sql(self) -> str:
        # 顺手 JOIN 出正文,省掉「先查命中、再回表取文本」的第二次往返
        return f"""
            SELECT e.article_id, e.ordinal,
                   1 - (e.embedding <=> CAST(:embedding AS vector)) AS score,
                   c.content
            FROM {self.table} e
            JOIN knowledge_chunks c
              ON c.article_id = e.article_id AND c.ordinal = e.ordinal
            WHERE e.embedder = :embedder AND e.dim = :dim
            ORDER BY e.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """

    def drop_sql(self) -> str:
        return f"DELETE FROM {self.table} WHERE article_id = :article_id"

    async def ensure_schema(self, session: AsyncSession) -> None:
        for statement in self.schema_statements():
            await session.execute(text(statement))
        await session.commit()

    @staticmethod
    def _literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

    async def upsert_chunks(
        self, session: AsyncSession, records: Sequence[ChunkVectorRecord]
    ) -> int:
        statement = text(self.upsert_sql())
        for record in records:
            await session.execute(
                statement,
                {
                    "article_id": record.article_id,
                    "ordinal": record.ordinal,
                    "embedder": record.embedder,
                    "dim": record.dim,
                    "embedding": self._literal(record.vector),
                },
            )
        await session.commit()
        return len(records)

    async def search_chunks(
        self,
        session: AsyncSession,
        *,
        vector: list[float],
        embedder: str,
        dim: int,
        limit: int,
    ) -> list[ChunkHit]:
        rows = await session.execute(
            text(self.search_sql()),
            {
                "embedding": self._literal(vector),
                "embedder": embedder,
                "dim": dim,
                "limit": limit,
            },
        )
        return [
            ChunkHit(
                article_id=row[0], ordinal=row[1], score=float(row[2]), content=row[3] or ""
            )
            for row in rows
        ]

    async def drop_article_chunks(self, session: AsyncSession, *, article_id: int) -> None:
        await session.execute(text(self.drop_sql()), {"article_id": article_id})


def build_vector_store(settings: Settings) -> VectorStore | None:
    backend = settings.vector_backend.strip().lower()
    if backend == "none":
        return None

    if backend == "pgvector" or (backend == "auto" and settings.is_postgres):
        if not settings.is_postgres:
            logger.warning(
                "VECTOR_BACKEND=pgvector 但当前不是 Postgres,已关闭向量检索",
                extra={"event": "retrieval.backend_mismatch"},
            )
            return None
        return PgVectorStore(dim=settings.embedding_dim)

    return SqliteVectorStore()


def build_chunk_vector_store(
    settings: Settings, *, embedder: str
) -> ChunkVectorStore | None:
    """段落级向量后端。和文章级同一套选择规则,关掉向量检索时返回 None。

    需要 embedder 名字,是因为 Postgres 的索引要按 embedder 分区建。
    """
    backend = settings.vector_backend.strip().lower()
    if backend == "none":
        return None

    if backend == "pgvector" or (backend == "auto" and settings.is_postgres):
        if not settings.is_postgres:
            logger.warning(
                "VECTOR_BACKEND=pgvector 但当前不是 Postgres,已关闭段落级向量检索",
                extra={"event": "retrieval.backend_mismatch"},
            )
            return None
        return PgChunkVectorStore(dim=settings.embedding_dim, embedder=embedder)

    return SqliteChunkVectorStore()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K
) -> dict[int, float]:
    """把多路排名融合成一个分数字典:第 rank 名贡献 1/(k+rank),同一条相加。"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, article_id in enumerate(ranking, start=1):
            scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_rankings(
    keyword_ids: Sequence[int],
    vector_hits: Sequence[tuple[int, float]],
) -> list[RankedArticle]:
    """融合关键词与向量两路排名,返回按融合分倒序的结果。"""
    vector_ids = [article_id for article_id, _ in vector_hits]
    scores = reciprocal_rank_fusion([list(keyword_ids), vector_ids])

    keyword_rank = {article_id: rank for rank, article_id in enumerate(keyword_ids, start=1)}
    vector_rank = {article_id: rank for rank, article_id in enumerate(vector_ids, start=1)}
    vector_score = dict(vector_hits)

    ranked = [
        RankedArticle(
            article_id=article_id,
            fusion_score=score,
            keyword_rank=keyword_rank.get(article_id),
            vector_rank=vector_rank.get(article_id),
            vector_score=vector_score.get(article_id),
        )
        for article_id, score in scores.items()
    ]
    ranked.sort(key=lambda item: (-item.fusion_score, item.article_id))
    return ranked


def chunk_hits_to_article_ranking(hits: Sequence[ChunkHit]) -> list[tuple[int, float]]:
    """段落命中折算成文章排名:同一篇取「最好的那一块」的分数。

    刻意不做多块求和:长文档块多,求和会让它天然占优,排序就变成「谁字多谁靠前」。
    """
    best: dict[int, float] = {}
    for hit in hits:
        if hit.score > best.get(hit.article_id, -1.0):
            best[hit.article_id] = hit.score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def merge_vector_rankings(
    article_ranking: Sequence[tuple[int, float]],
    chunk_ranking: Sequence[tuple[int, float]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """把「整篇向量」与「段落向量」两路合成一路向量召回。

    两路分数不可比(一个算整篇、一个只算最好的一段),所以仍然按排名融合;
    展示用的分数取两路里更高的那个,排序则看 RRF 的融合分。
    """
    if not chunk_ranking:
        return list(article_ranking)
    if not article_ranking:
        return list(chunk_ranking)

    scores = reciprocal_rank_fusion(
        [
            [article_id for article_id, _ in article_ranking],
            [article_id for article_id, _ in chunk_ranking],
        ],
        k=k,
    )
    best: dict[int, float] = {}
    for article_id, score in (*article_ranking, *chunk_ranking):
        best[article_id] = max(best.get(article_id, 0.0), score)

    ordered = sorted(scores, key=lambda article_id: (-scores[article_id], article_id))
    return [(article_id, best[article_id]) for article_id in ordered]
