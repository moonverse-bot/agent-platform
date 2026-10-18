"""建表 + 灌入种子数据 + 回填知识库向量(幂等,重复启动不会重复插入)。"""

import logging

from sqlalchemy import delete, or_, select

from app.agent.chunking import ChunkingSettings, split_document
from app.agent.embeddings import Embedder, build_embedder, embed_cached
from app.agent.retrieval import (
    ChunkVectorRecord,
    VectorRecord,
    build_chunk_vector_store,
    build_vector_store,
)
from app.config import get_settings
from app.core.security import hash_password
from app.db.base import SessionLocal, create_all
from app.db.models import (
    Conversation,
    KnowledgeArticle,
    KnowledgeChunk,
    Message,
    Ticket,
    User,
)

logger = logging.getLogger(__name__)

SEED_ARTICLES: list[dict[str, str]] = [
    {
        "title": "如何重置账号密码",
        "content": (
            "在登录页点击「忘记密码」,输入注册邮箱后系统会发送重置链接,链接 30 分钟内有效。"
            "若长时间未收到邮件,请检查垃圾邮件目录;连续 3 次未收到可联系人工客服。"
        ),
        "tags": "账号,密码,登录",
    },
    {
        "title": "工单处理时效说明",
        "content": (
            "普通工单承诺 24 小时内首次响应,高优先级工单为 4 小时,紧急工单为 1 小时。"
            "工单超过承诺时间未响应会自动升级并通知值班负责人。"
        ),
        "tags": "工单,时效,升级",
    },
    {
        "title": "发票申请流程",
        "content": (
            "在「费用中心 - 发票管理」提交开票申请,需填写公司名称、税号与收件邮箱。"
            "电子发票通常在 1-3 个工作日开出,纸质发票另需 5 个工作日邮寄。"
        ),
        "tags": "发票,财务,开票",
    },
    {
        "title": "退款政策",
        "content": (
            "自购买之日起 7 天内支持无理由退款,超过 7 天按剩余服务周期折算。"
            "已开具发票的订单需先退回发票原件方可退款,到账时间为 3-5 个工作日。"
        ),
        "tags": "退款,财务,售后",
    },
    {
        "title": "API 调用频率限制",
        "content": (
            "默认套餐每分钟 60 次请求、每天 10000 次;企业版可通过工单申请提升配额。"
            "超出限制会返回 429 状态码,建议客户端做退避重试。"
        ),
        "tags": "API,限流,技术",
    },
    {
        "title": "如何导出历史数据",
        "content": (
            "在「设置 - 数据管理」中选择时间范围后导出 CSV。单次导出上限 10 万条,"
            "超出请分批导出或提交工单申请离线导出。"
        ),
        "tags": "数据,导出,报表",
    },
    {
        # 这篇刻意写得长、且分了好几段:切块与段落级检索的价值在长文档上才看得出来
        "title": "企业版服务与结算说明",
        "content": (
            "企业版按年签约,合同期内可随时追加席位,追加部分按剩余月份折算费用;"
            "减少席位不退款,只能等到下一年度续约时生效。\n"
            "结算周期为自然月,每月 1 日出账,账单在 3 个工作日内推送到合同预留邮箱。"
            "对公转账到账后系统自动核销,若超过 7 个工作日仍未核销,请带付款凭证提交工单人工核对。\n"
            "发票按账单金额开具,可申请纸质专用发票,邮寄时间比电子发票多 5 个工作日;"
            "发票抬头变更需要在开票前提交,已开出的发票只能先退回再重开。\n"
            "服务到期前 30 天系统会发送续费提醒;到期后数据保留 90 天,"
            "这期间续费可原样恢复,超过 90 天数据会被清理且无法找回。\n"
            "SLA 按季度统计可用性,低于 99.9% 的部分按合同约定比例抵扣下期费用,"
            "抵扣金额会在下一期账单中自动体现,不需要额外申请。"
        ),
        "tags": "企业版,结算,合同",
    },
    {
        "title": "API 接入与限流调优指南",
        "content": (
            "接口鉴权统一使用 Bearer Token。Token 在「设置 - 开放平台」创建,默认有效期 90 天,"
            "过期前 7 天会通过邮件与站内信提醒;同一账号最多保留 5 个有效 Token,"
            "泄露的 Token 应立即吊销,吊销后 1 分钟内全网生效,期间已发出的请求不再校验。\n"
            "请求失败请先看响应体的 error.code:429 表示触发限流,RATE_LIMIT_EXCEEDED 表示超出配额,"
            "两者都建议按指数退避重试,首次退避 1 秒、之后每次翻倍、最多重试 5 次;"
            "5xx 多为服务端瞬时抖动,同样适合重试,而参数错误这类 4xx 重试没有意义,"
            "只会白白消耗配额,应当先修参数再调用。\n"
            "批量场景优先使用异步接口:提交任务后拿到 task_id,再轮询任务状态或订阅回调通知,"
            "单个任务上限 1000 条,超出请拆成多个任务;同步接口单次最多 100 条。\n"
            "需要更高配额时提交工单,注明调用场景、峰值 QPS 与预估日调用量,"
            "官方承诺 3 个工作日内答复;企业版客户可通过客户成功经理加急处理。"
        ),
        "tags": "API,限流,接入",
    },
]


# 演示账号:前端登录页默认就填这组信息,种子数据保证它开箱即用
DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "password-123"

DEMO_TICKETS: list[dict[str, str]] = [
    {
        "title": "登录后台偶发 500 报错",
        "description": "点击登录按钮后偶尔返回 500,刷新一次又正常,大约每天出现一两次。",
        "status": "open",
        "priority": "high",
    },
    {
        "title": "发票抬头需要修改",
        "description": "公司名称变更,想把已开出的发票抬头改成新的。",
        "status": "processing",
        "priority": "normal",
    },
    {
        "title": "申请提升 API 调用配额",
        "description": "业务量上来了,每分钟 60 次不够用,希望提到 300 次。",
        "status": "closed",
        "priority": "urgent",
    },
]

# 一段预置对话,让左侧会话列表不是空的
DEMO_HISTORY: list[tuple[str, str]] = [
    (
        "退款政策是什么?",
        "自购买之日起 7 天内支持无理由退款,超过 7 天按剩余服务周期折算。"
        "已开具发票的订单需要先退回发票原件,款项 3-5 个工作日到账。",
    ),
    (
        "我有哪些工单?",
        "你目前有 3 条工单:#1 登录后台偶发 500 报错(待处理)、"
        "#2 发票抬头需要修改(处理中)、#3 申请提升 API 调用配额(已关闭)。",
    ),
]


async def seed_demo_user() -> bool:
    """创建演示账号及其示例数据,返回本次是否新建。

    先按邮箱查一次再插入,所以重复启动不会重复灌数据;也已经注册过的库不会被动。
    """
    async with SessionLocal() as session:
        existing = await session.scalar(select(User.id).where(User.email == DEMO_EMAIL))
        if existing is not None:
            return False

        user = User(
            email=DEMO_EMAIL,
            display_name="演示账号",
            hashed_password=hash_password(DEMO_PASSWORD),
        )
        session.add(user)
        await session.flush()  # 先拿到 user.id,下面建工单和会话都要用

        session.add_all([Ticket(owner_id=user.id, **item) for item in DEMO_TICKETS])

        conversation = Conversation(owner_id=user.id, title=DEMO_HISTORY[0][0])
        session.add(conversation)
        await session.flush()
        session.add_all(
            [
                Message(conversation_id=conversation.id, role=role, content=content)
                for question, answer in DEMO_HISTORY
                for role, content in (("user", question), ("assistant", answer))
            ]
        )

        await session.commit()
        return True


async def seed_knowledge() -> int:
    """插入知识库里还缺的种子文章,返回本次新增条数。

    按标题判断是否已经存在,所以「重复启动不会重复插入」依旧成立;
    而往 SEED_ARTICLES 里新增文章时,已有的库重启一次就能补上,
    不需要手工删库重灌。
    """
    async with SessionLocal() as session:
        existing = set(await session.scalars(select(KnowledgeArticle.title)))
        missing = [item for item in SEED_ARTICLES if item["title"] not in existing]
        if not missing:
            return 0
        session.add_all([KnowledgeArticle(**item) for item in missing])
        await session.commit()
        return len(missing)


async def init_db() -> None:
    await create_all()
    inserted = await seed_knowledge()
    demo_created = await seed_demo_user() if get_settings().seed_demo_data else False
    indexed = await refresh_knowledge_index()
    logger.info(
        "数据库初始化完成,新增知识库文章 %s 条,演示账号 %s,新增向量 %s 条",
        inserted,
        "已创建" if demo_created else "已存在/未启用",
        indexed,
        extra={"event": "db.init"},
    )


def _index_text(article: KnowledgeArticle) -> str:
    """一篇文档的向量取自标题 + 标签 + 正文,标题和标签对召回的影响因此被放大。"""
    return f"{article.title}\n{article.tags}\n{article.content}"


async def refresh_knowledge_index(force: bool = False) -> int:
    """回填知识库索引(整篇 + 段落),返回文章级向量本次写入的条数。

    这是整个检索链路的「建索引」入口,启动时调用一次即可。两级的缓存判断互相独立:
    换 Embedding 模型会同时触发两级重算,而调切块参数只会影响段落那一级。
    """
    written = await _refresh_article_vectors(force=force)
    chunks = await refresh_knowledge_chunks(force=force)
    if chunks:
        logger.info(
            "知识库段落向量已更新 %s 块",
            chunks,
            extra={"event": "retrieval.chunk_index"},
        )
    return written


async def _refresh_article_vectors(force: bool = False) -> int:
    """给还没有向量的知识库文章补算「整篇」向量,返回本次写入的条数。

    - 幂等:已经有当前 embedder + 维度的向量的文章会被跳过;
    - 换 Embedding 模型或维度之后,旧向量自动视为过期并重算(article_id 是主键,直接覆盖);
    - 关掉向量检索、或向量化失败时返回 0,不影响关键词检索这条主线。
    """
    settings = get_settings()
    store = build_vector_store(settings)
    if store is None:
        return 0

    embedder = build_embedder(settings)
    async with SessionLocal() as session:
        try:
            await store.ensure_schema(session)
        except Exception:  # noqa: BLE001 - 建不了索引就退化成纯关键词检索
            logger.exception(
                "向量索引初始化失败,本次跳过向量检索",
                extra={"event": "retrieval.schema_error"},
            )
            return 0

        indexed = await store.indexed_ids(session, embedder=embedder.name, dim=embedder.dim)
        articles = list(await session.scalars(select(KnowledgeArticle)))
        pending = [article for article in articles if force or article.id not in indexed]
        if not pending:
            return 0

        try:
            vectors = await embed_cached(embedder, [_index_text(article) for article in pending])
        except Exception:  # noqa: BLE001 - 模型服务抖动不该让服务起不来
            logger.exception(
                "知识库向量计算失败,本次跳过",
                extra={"event": "retrieval.embed_error"},
            )
            return 0

        await store.upsert(
            session,
            [
                VectorRecord(
                    article_id=article.id,
                    embedder=embedder.name,
                    dim=embedder.dim,
                    vector=vector,
                )
                for article, vector in zip(pending, vectors)
            ],
        )
        logger.info(
            "知识库向量已更新 %s 条",
            len(pending),
            extra={"event": "retrieval.index", "embedder": embedder.name},
        )
        return len(pending)


def _chunks_up_to_date(
    current: list[KnowledgeChunk],
    wanted: list[str],
    embedder: Embedder,
    force: bool,
) -> bool:
    """这篇文档的段落是否已经是最新的:正文对得上,且每块都有当前模型算的向量。"""
    if force or [row.content for row in current] != wanted:
        return False
    return all(
        row.embedder == embedder.name and row.dim == embedder.dim and row.vector
        for row in current
    )


async def refresh_knowledge_chunks(force: bool = False) -> int:
    """切块并补算段落向量,返回本次写入的向量条数。

    - 幂等:段落正文与当前切块参数一致、且向量是当前 embedder + 维度时整篇跳过;
    - 换切块大小/重叠、改过正文、换 Embedding 模型或维度,都会让这一篇的段落整篇重建
      (块数变了、顺序也会变,直接重建比逐块对齐更省心,知识库本来就不大);
    - 关掉切块、关掉向量检索、或向量化失败时返回 0,不影响关键词检索这条主线。
    """
    settings = get_settings()
    if not settings.chunking_enabled:
        return 0

    chunking = ChunkingSettings(
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )
    embedder = build_embedder(settings)
    store = build_chunk_vector_store(settings, embedder=embedder.name)
    if store is None:
        return 0

    async with SessionLocal() as session:
        try:
            await store.ensure_schema(session)
        except Exception:  # noqa: BLE001 - 建不了索引就退化成纯关键词检索
            logger.exception(
                "段落向量索引初始化失败,本次跳过",
                extra={"event": "retrieval.chunk_schema_error"},
            )
            return 0

        articles = list(await session.scalars(select(KnowledgeArticle)))
        stored = list(await session.scalars(select(KnowledgeChunk)))
        by_article: dict[int, list[KnowledgeChunk]] = {}
        for row in stored:
            by_article.setdefault(row.article_id, []).append(row)

        stale: list[KnowledgeArticle] = []
        for article in articles:
            wanted = split_document(article.title, article.content, chunking)
            current = sorted(by_article.get(article.id, []), key=lambda row: row.ordinal)
            if not _chunks_up_to_date(current, wanted, embedder, force):
                stale.append(article)

        if not stale:
            return 0

        for article in stale:
            # 先撤掉这一篇的旧段落(正文与向量一起),再按当前切块参数重建
            await session.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.article_id == article.id)
            )
            await store.drop_article_chunks(session, article_id=article.id)
            await session.flush()
            session.add_all(
                KnowledgeChunk(article_id=article.id, ordinal=index, content=text)
                for index, text in enumerate(split_document(article.title, article.content, chunking))
            )
        await session.commit()

        pending = list(
            await session.scalars(
                select(KnowledgeChunk).where(
                    or_(
                        KnowledgeChunk.embedder != embedder.name,
                        KnowledgeChunk.dim != embedder.dim,
                        KnowledgeChunk.vector == "",
                    )
                )
            )
        )
        if not pending:
            return 0

        try:
            vectors = await embed_cached(embedder, [row.content for row in pending])
        except Exception:  # noqa: BLE001 - 模型服务抖动不该让服务起不来
            logger.exception(
                "段落向量计算失败,本次跳过",
                extra={"event": "retrieval.chunk_embed_error"},
            )
            return 0

        return await store.upsert_chunks(
            session,
            [
                ChunkVectorRecord(
                    article_id=row.article_id,
                    ordinal=row.ordinal,
                    embedder=embedder.name,
                    dim=embedder.dim,
                    vector=vector,
                )
                for row, vector in zip(pending, vectors)
            ],
        )
