"""文本向量化:统一走 OpenAI 兼容的 /embeddings 协议,没配 Key 时用离线实现兜底。

在线实现可以直接对接 OpenAI、硅基流动(SiliconFlow)、智谱、阿里云 DashScope 等
任何提供 OpenAI 兼容 embeddings 接口的服务,换服务只改配置、不改代码。

离线兜底 `HashingEmbedder` **不是神经语义向量**:它把字符二元组做特征哈希,
只能捕捉「字形/词形重合」的相似度,质量大致相当于关键词匹配。它存在的意义是
让没有 Embedding Key 的本地开发和测试也能跑通整条向量检索链路,
所以结果里会带上 embedder 名字,避免把兜底结果误当成真实语义召回。

切词统一走 `app/agent/text.py` —— 关键词打分、离线向量、重排用的是同一套词,
否则同一次检索里三处对「哪些是词」的理解会悄悄分叉。
"""

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Protocol

import httpx

from app.agent.text import terms
from app.config import Settings

logger = logging.getLogger(__name__)

_CACHE_MAX = 512
# (embedder 名字, 原文) -> 向量。同一句问题反复问时不用再算一次,省 token 也省延迟。
_CACHE: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()


class Embedder(Protocol):
    name: str
    dim: int
    #: 该 embedder 下「算命中」的建议最低余弦。不同模型的分数量纲差别很大,
    #: 所以阈值跟着实现走,而不是写一个全局常量。
    default_min_score: float

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """真实余弦相似度。在线服务返回的向量不保证归一化,所以这里不能只算点积。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class HashingEmbedder:
    """离线兜底:统一分词 + 特征哈希 + 带符号累加 + L2 归一化。零依赖、确定性。"""

    # 名字里带版本号:切词规则一变,同一段文本算出来的向量就变了,
    # 而索引是以「embedder 名字 + 维度」作为版本的。改名能保证旧向量被自动重算,
    # 不会出现「一半新规则、一半旧规则」的混合索引。
    name = "local-hash-bigram-v2"
    # 实测(1024 维、当前知识库):不相关文本余弦 <= 0.06,真实命中 0.13~0.27。
    # 阈值取 0.12 能把噪声挡在外面,又不会误杀正确结果。
    default_min_score = 0.12

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in terms(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            # 带符号累加:两个不同 token 撞进同一个桶时有机会互相抵消,减少哈希碰撞的偏差
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        return l2_normalize(vector)


class OpenAICompatibleEmbedder:
    """任意 OpenAI 兼容的 /embeddings 服务。"""

    # 神经 embedding 的余弦普遍偏高:不相关也有 0.2~0.35,相关通常 0.5 以上
    default_min_score = 0.3

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 30.0,
    ) -> None:
        self.name = f"openai-compatible:{model}"
        self.dim = dim
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self._model, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings", json=payload, headers=headers
            )
            response.raise_for_status()
            body = response.json()

        rows = sorted(body.get("data", []), key=lambda row: row.get("index", 0))
        vectors = [[float(value) for value in row["embedding"]] for row in rows]
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"Embedding 维度不匹配:接口返回 {len(vectors[0])} 维,"
                f"配置的 EMBEDDING_DIM 是 {self.dim}。请改成一致的值并重建向量索引。"
            )
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    if settings.online_embeddings_enabled:
        if not settings.embedding_api_key.strip():
            logger.warning(
                "EMBEDDING_PROVIDER=openai 但没有配 EMBEDDING_API_KEY,退回离线哈希向量",
                extra={"event": "embedding.fallback"},
            )
        else:
            return OpenAICompatibleEmbedder(
                api_key=settings.embedding_api_key.strip(),
                base_url=settings.embedding_base_url.strip() or "https://api.openai.com/v1",
                model=settings.embedding_model.strip() or "text-embedding-3-small",
                dim=settings.embedding_dim,
                timeout=settings.embedding_timeout_seconds,
            )

    return HashingEmbedder(dim=settings.embedding_dim)


async def embed_cached(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """带查询缓存的向量化:相同文本 + 相同 embedder 只算一次。"""
    ordered = list(dict.fromkeys(texts))
    missing = [text for text in ordered if (embedder.name, text) not in _CACHE]

    if missing:
        vectors = await embedder.embed(missing)
        for text, vector in zip(missing, vectors):
            _CACHE[(embedder.name, text)] = vector
            _CACHE.move_to_end((embedder.name, text))
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)

    return [_CACHE[(embedder.name, text)] for text in texts]


def clear_embedding_cache() -> None:
    _CACHE.clear()
