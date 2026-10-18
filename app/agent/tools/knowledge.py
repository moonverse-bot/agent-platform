"""知识库检索工具:关键词 + 向量(整篇 / 段落)召回,再重排。

三路信号各管一类问题,谁也替代不了谁:

  * 关键词(keyword):字面命中。工单编号、产品型号、专有名词这类必须精确匹配的查询
    只能靠它;而且零依赖、结果可解释,是系统最后的兜底。
  * 整篇向量(vector):换个说法也能召回。用户问「钱怎么退回来」而文档写的是「退款」,
    关键词一个词都命中不了,向量能拉回来。
  * 段落向量(chunks):同一套向量在「段落」粒度上再算一遍。文档越长,整篇向量被
    稀释得越厉害;切块之后,只要有一小段正好在讲这件事就能精确命中,而且会把
    命中的那一段作为 matched_chunk 返回 —— 给模型的是最贴题的几句话,而不是整篇。

召回结果先按 RRF 融合(app/agent/retrieval.py),再交给重排器重排(app/agent/rerank.py)。
任何一路不可用(没配 Embedding、索引没建、模型服务报错)都会自动退化成别的路,
所以这个工具在零配置下也能正常工作。
"""

import json
import logging

from sqlalchemy import select

from app.agent.embeddings import Embedder, build_embedder, embed_cached
from app.agent.rerank import build_reranker
from app.agent.retrieval import (
    ChunkHit,
    RankedArticle,
    build_chunk_vector_store,
    build_vector_store,
    chunk_hits_to_article_ranking,
    fuse_rankings,
    merge_vector_rankings,
)
from app.agent.text import normalize, terms
from app.agent.tools.base import Tool, ToolContext
from app.config import Settings, get_settings
from app.core.breaker import CircuitBreaker
from app.db.models import KnowledgeArticle

logger = logging.getLogger(__name__)

_TITLE_WEIGHT = 3
_TAG_WEIGHT = 2
_CONTENT_WEIGHT = 1
# 关键词一路最多送这么多候选进融合,避免长尾噪声把 RRF 的排名压扁
_MAX_KEYWORD_CANDIDATES = 50
# 回给模型的命中片段上限。段落本来就远小于这个数,这里只是防御性截断。
_CHUNK_PREVIEW_LIMIT = 500

# 向量这一路失败后的冷却时间。Embedding 服务挂了如果每次提问都先等一次超时,
# 一个挂掉的外部依赖会把所有对话都拖慢;冷却期内直接跳过向量检索。
_VECTOR_BREAKER = CircuitBreaker(cooldown_seconds=60.0)


def reset_vector_circuit() -> None:
    """立刻恢复向量检索(测试用,也方便运维排查完马上打开)。"""
    _VECTOR_BREAKER.reset()


def _keyword_ranking(articles: list[KnowledgeArticle], query_terms: list[str]) -> list[int]:
    """按关键词得分倒序返回文章 id,得分为 0 的不进候选。

    比对前把文档字段统一小写、把标点折成空格:切出来的英文词是小写的,
    文档里写的「API」得先折成「api」才可能命中。
    """
    scored: list[tuple[int, int]] = []
    for article in articles:
        title = normalize(article.title or "")
        tags = normalize(article.tags or "")
        content = normalize(article.content or "")
        score = 0
        for term in query_terms:
            if term in title:
                score += _TITLE_WEIGHT
            if term in tags:
                score += _TAG_WEIGHT
            if term in content:
                score += _CONTENT_WEIGHT
        scored.append((score, article.id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [article_id for score, article_id in scored if score > 0][:_MAX_KEYWORD_CANDIDATES]


async def _vector_recall(
    ctx: ToolContext, query: str, settings: Settings
) -> tuple[list[tuple[int, float]], str, dict[int, ChunkHit]]:
    """向量召回。返回 (文章排名, 状态说明, 每篇命中的那一段)。

    这里刻意把所有异常都吞掉并转成状态说明:向量检索是「增强」而不是「主链路」,
    它挂了也绝不能让 search_knowledge 跟着挂。
    """
    if not settings.vector_search_enabled:
        return [], "mode=keyword", {}

    embedder = build_embedder(settings)
    article_store = build_vector_store(settings)
    chunk_store = (
        build_chunk_vector_store(settings, embedder=embedder.name)
        if settings.chunking_enabled
        else None
    )
    if article_store is None and chunk_store is None:
        return [], "vector-backend=none", {}

    if _VECTOR_BREAKER.is_open:
        return [], "vector-circuit-open", {}

    try:
        vector = (await embed_cached(embedder, [query]))[0]
    except Exception:  # noqa: BLE001
        logger.exception("查询向量化失败,退化到关键词检索")
        _VECTOR_BREAKER.trip()
        return [], "embedding-failed", {}

    article_ranking: list[tuple[int, float]] = []
    chunk_ranking: list[tuple[int, float]] = []
    matched: dict[int, ChunkHit] = {}
    chunk_hits = 0

    try:
        if article_store is not None:
            article_ranking = await article_store.search(
                ctx.session,
                vector=vector,
                embedder=embedder.name,
                dim=embedder.dim,
                limit=settings.retrieval_candidates,
            )
        if chunk_store is not None:
            hits = await chunk_store.search_chunks(
                ctx.session,
                vector=vector,
                embedder=embedder.name,
                dim=embedder.dim,
                limit=settings.retrieval_candidates,
            )
            chunk_hits = len(hits)
            chunk_ranking = chunk_hits_to_article_ranking(hits)
            for hit in hits:
                matched.setdefault(hit.article_id, hit)
    except Exception:  # noqa: BLE001
        logger.exception("向量检索失败,退化到关键词检索")
        _VECTOR_BREAKER.trip()
        return [], "search-failed", {}

    store = article_store or chunk_store
    label = f"{store.name}:{embedder.name}" if store else f"vector:{embedder.name}"
    # 低于阈值的不算命中:语义相近的负样本余弦也不会太低,不设阈值就会把不相关的
    # 文档塞进上下文。阈值默认跟着 embedder 走,配置里显式给了正数才覆盖。
    min_score = settings.retrieval_min_vector_score or embedder.default_min_score
    article_ranking = [item for item in article_ranking if item[1] >= min_score]
    chunk_ranking = [item for item in chunk_ranking if item[1] >= min_score]
    if not article_ranking and not chunk_ranking:
        return [], f"{label} (无命中,阈值 {min_score:g})", {}

    merged = merge_vector_rankings(article_ranking, chunk_ranking)
    scope = f"整篇 {len(article_ranking)} 篇"
    if chunk_store is not None:
        scope += f" + 段落 {chunk_hits} 段"

    # 被阈值挡掉的文章,它的命中片段也不该再被引用
    kept = {article_id for article_id, _ in merged}
    matched = {article_id: hit for article_id, hit in matched.items() if article_id in kept}
    return merged, f"{label} (阈值 {min_score:g};{scope})", matched


def _rerank_text(article: KnowledgeArticle | None, hit: ChunkHit | None) -> str:
    """重排看哪段文本:段落级召回就看重排那一段,否则看整篇。

    命中的是「那一小段」,重排再去看整篇会把刚拿到的精度又摊薄回去,
    所以重排与召回保持同一粒度。
    """
    if hit is not None:
        return hit.content
    if article is None:
        return ""
    return f"{article.title}\n{article.tags}\n{article.content}"


async def _rerank(
    query: str,
    settings: Settings,
    candidates: list[RankedArticle],
    by_id: dict[int, KnowledgeArticle],
    matched: dict[int, ChunkHit],
) -> tuple[list[tuple[RankedArticle, float | None]], str | None]:
    """对头部候选重排一次,返回 (结果, 重排器名字)。

    重排永远是「锦上添花」:关掉了、候选只有一条、服务报错,都直接沿用召回顺序。
    """
    limited = candidates[: max(1, settings.rerank_candidate_limit)]
    reranker = build_reranker(settings)
    if reranker is None or len(limited) < 2:
        return [(item, None) for item in limited], None

    documents = [
        _rerank_text(by_id.get(item.article_id), matched.get(item.article_id))
        for item in limited
    ]
    try:
        hits = await reranker.rerank(query, documents, top_n=len(limited))
    except Exception:  # noqa: BLE001
        logger.exception("重排失败,保留召回顺序")
        return [(item, None) for item in limited], None

    # 重排分相同时回到召回顺序(融合分高的在前),而不是「谁的 id 小谁靠前」——
    # 后者会让同样的输入在不同数据上给出看起来随机的结果。
    ranked_hits = sorted(
        hits, key=lambda item: (-item[1], -limited[item[0]].fusion_score, item[0])
    )
    ordered: list[tuple[RankedArticle, float | None]] = [
        (limited[index], score) for index, score in ranked_hits
    ]
    covered = {index for index, _ in hits}
    # 有些服务只返回 top_n 条,没被覆盖的候选按原顺序跟在后面,一条都不丢
    ordered.extend(
        (item, None) for index, item in enumerate(limited) if index not in covered
    )
    return ordered, reranker.name


async def search_knowledge(ctx: ToolContext, args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "检索失败:query 不能为空。"

    query_terms = terms(query)
    if not query_terms:
        return "检索失败:query 中没有可用于检索的内容。"

    settings = get_settings()
    articles = list(await ctx.session.scalars(select(KnowledgeArticle)))
    if not articles:
        return "知识库为空,建议让用户提交工单由人工跟进。"

    keyword_ids = (
        _keyword_ranking(articles, query_terms) if settings.keyword_search_enabled else []
    )
    vector_hits, vector_note, matched = await _vector_recall(ctx, query, settings)

    if not settings.keyword_search_enabled and not vector_hits:
        # 配置成「只走向量」但向量这一路不可用时,不要让工具直接哑掉
        keyword_ids = _keyword_ranking(articles, query_terms)
        vector_note += " -> 回退关键词"

    candidates = fuse_rankings(keyword_ids, vector_hits)
    if not candidates:
        return "知识库中没有找到相关内容,可以建议用户提交工单由人工跟进。"

    by_id = {article.id: article for article in articles}
    ordered, reranker_name = await _rerank(query, settings, candidates, by_id, matched)

    results = []
    for item, rerank_score in ordered[: max(1, settings.retrieval_top_k)]:
        article = by_id.get(item.article_id)
        if article is None:
            continue
        hit = matched.get(item.article_id)
        results.append(
            {
                "title": article.title,
                "content": article.content,
                "tags": article.tags,
                "fusion_score": round(item.fusion_score, 5),
                "keyword_rank": item.keyword_rank,
                "vector_rank": item.vector_rank,
                "vector_score": (
                    round(item.vector_score, 4) if item.vector_score is not None else None
                ),
                # 命中的原文片段:段落级召回才有,比整篇正文更贴题
                "matched_chunk": (
                    hit.content[:_CHUNK_PREVIEW_LIMIT] if hit is not None else None
                ),
                "rerank_score": (
                    round(rerank_score, 4) if rerank_score is not None else None
                ),
            }
        )

    return json.dumps(
        {
            "query": query,
            "retrieval_mode": settings.retrieval_mode,
            "vector": vector_note,
            "reranker": reranker_name,
            "results": results,
        },
        ensure_ascii=False,
    )


KNOWLEDGE_TOOLS = [
    Tool(
        name="search_knowledge",
        description=(
            "在客服知识库中检索资料,返回最相关的若干篇文章(命中段落会一并给出)。"
            "检索会把关键词命中、整篇向量与段落向量召回的结果融合排序并重排,"
            "所以用用户的原话提问即可,不必自己提炼关键词。"
            "回答产品政策、流程、费用等问题前应先调用它,不要凭记忆作答。"
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "用户的原始问题或关键词"}},
            "required": ["query"],
        },
        handler=search_knowledge,
        tags=["knowledge"],
    ),
]
