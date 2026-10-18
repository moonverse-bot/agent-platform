"""重排:召回之后再按「和问题有多相关」重新排一遍。

召回为了快,用的是粗筛(关键词 + 向量),给出的顺序是近似的;重排这一层看得更细,
但更贵,所以只对头部少量候选做,性价比最高。

默认实现是**词法覆盖度**:统计问题里的词有多少出现在候选文本里(0~1 的分数),
覆盖度相同时再看这些词一共出现了多少次 —— 同一个词反复出现,说明候选确实围绕这件事。
它零依赖、确定性强、分数可解释,但它**不是神经网络 cross-encoder** ——
它一样理解不了同义词,所以工具输出里一定会带上重排器的名字,
别把它的分数当成语义相关性。真正的语义重排需要一个重排模型,因此这里同时提供了
OpenAI 兼容 /rerank(Cohere、硅基流动等都用这套协议)的实现,改配置即可切换。
"""

from collections.abc import Sequence
from typing import Protocol

import httpx

from app.agent.text import normalize, unique_terms
from app.config import Settings

#: (候选在入参 documents 里的下标, 相关度分数)
RerankHit = tuple[int, float]


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankHit]: ...


class LexicalReranker:
    """零依赖的词法覆盖度重排。默认实现,离线可用。"""

    name = "lexical-coverage"

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankHit]:
        top_n = max(1, top_n)
        query_terms = unique_terms(query)
        if not query_terms:
            # 问题里一个词都切不出来(比如纯符号),重排没有依据,保持召回顺序
            return [(index, 0.0) for index in range(min(top_n, len(documents)))]

        scored: list[tuple[float, int, int]] = []
        for index, document in enumerate(documents):
            text = normalize(document)
            covered = sum(1 for term in query_terms if term in text)
            occurrences = sum(text.count(term) for term in query_terms)
            scored.append((covered / len(query_terms), occurrences, index))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [(index, score) for score, _, index in scored[:top_n]]


class OpenAICompatibleReranker:
    """任意 OpenAI 兼容的 /rerank 服务(Cohere、硅基流动、Jina 等都是这个协议)。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 10.0,
    ) -> None:
        self.name = f"rerank-api:{model}"
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankHit]:
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": list(documents),
            "top_n": max(1, top_n),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/rerank", json=payload, headers=headers
            )
            response.raise_for_status()
            body = response.json()

        scored: list[RerankHit] = []
        for row in body.get("results", []):
            try:
                index = int(row["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= index < len(documents):
                continue
            raw_score = row.get("relevance_score", row.get("score", 0.0))
            scored.append((index, float(raw_score)))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(1, top_n)]


def build_reranker(settings: Settings) -> Reranker | None:
    """按配置挑一个重排器。返回 None 表示重排被关掉,调用方跳过这一步即可。"""
    if not settings.rerank_enabled:
        return None

    provider = settings.rerank_provider.strip().lower()
    if provider in {"api", "openai"} and settings.rerank_api_key.strip():
        return OpenAICompatibleReranker(
            api_key=settings.rerank_api_key.strip(),
            base_url=settings.rerank_base_url.strip() or "https://api.siliconflow.cn/v1",
            model=settings.rerank_model.strip() or "BAAI/bge-reranker-v2-m3",
            timeout=settings.rerank_timeout_seconds,
        )

    # 选了 api 但没给 Key,或者本来就是默认的 lexical:都退回零依赖实现
    return LexicalReranker()
