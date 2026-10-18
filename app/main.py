"""FastAPI 应用入口:装配路由、中间件、生命周期。"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from app.agent.embeddings import build_embedder
from app.agent.rerank import build_reranker
from app.agent.retrieval import build_chunk_vector_store, build_vector_store
from app.agent.runner import AgentRunner
from app.api import auth, chat, conversations, tickets
from app.config import get_settings
from app.core.docs import render_docs_html
from app.core.logging import get_trace_id, set_trace_id, setup_logging
from app.core.ratelimit import build_rate_limiter
from app.db.base import SessionLocal
from app.db.init_db import init_db
from app.db.models import KnowledgeChunk

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"

TITLE = "智能工单 / 客服 Agent 平台"

DESCRIPTION = """
用户可以**用大白话提问**,大模型会自己判断该调用哪个工具(查工单 / 建工单 / 检索知识库),
拿到结果后再组织成中文回答。接口分四组:

- **auth**:注册、登录、查询当前用户。
- **tickets**:工单的增删改查与统计,是真实的业务数据接口,也是 Agent 可以调用的底层能力。
- **chat**:Agent 对话入口,以 SSE 流式返回工具调用过程与回答。
- **conversations**:会话历史查询与删除,供前端恢复聊天记录。
- **ops**:运维探活。

### 怎么在这里直接试

1. 调 `POST /api/auth/register` 注册账号(邮箱随便填,密码至少 8 位)。
2. 调 `POST /api/auth/login` 拿到 `access_token`。
3. 点右上角**授权**,把 token 填进去(只填 token,不要带 Bearer 前缀)。
4. 调 `POST /api/tickets` 建一条工单,再调 `POST /api/chat` 问它「我有哪些工单」。

带锁图标的接口都需要先完成第 3 步。
""".strip()

TAGS_METADATA = [
    {
        "name": "auth",
        "description": "注册、登录与身份查询。除注册和登录外,其余接口都需要 Bearer Token。",
    },
    {
        "name": "tickets",
        "description": "工单业务接口。数据按登录用户隔离,查不到别人的工单。",
    },
    {
        "name": "chat",
        "description": "Agent 对话入口,以 SSE 流式返回工具调用过程与回答。",
    },
    {"name": "conversations", "description": "会话历史。数据同样按登录用户隔离。"},
    {"name": "ops", "description": "运维探活接口。"},
]


async def _count_indexed_chunks(embedder_name: str, dim: int) -> int | None:
    """统计已经建好向量的段落数。查不动就返回 None —— 探活不该因为统计而失败。"""
    try:
        async with SessionLocal() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(
                    KnowledgeChunk.embedder == embedder_name,
                    KnowledgeChunk.dim == dim,
                )
            )
            return int(count or 0)
    except Exception:  # noqa: BLE001
        logger.warning("统计段落向量条数失败", exc_info=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.debug)
    if settings.jwt_secret == "dev-only-secret-change-me-before-production":
        logger.warning(
            "JWT_SECRET 仍是代码里的默认值,公网部署前必须换成随机串",
            extra={"event": "app.insecure_jwt_secret"},
        )
    await init_db()
    app.state.rate_limiter = await build_rate_limiter(
        settings.rate_limit_per_minute, settings.redis_url
    )
    app.state.agent = AgentRunner(settings)
    logger.info(
        "应用启动完成",
        extra={"event": "app.startup", "model": settings.llm_model},
    )
    yield
    logger.info("应用关闭", extra={"event": "app.shutdown"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version="0.1.0",
        openapi_tags=TAGS_METADATA,
        docs_url=None,  # 关掉自带文档页,下面挂中文化的那个
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        # 支持调用方透传 trace_id,便于跨服务串联一次请求
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]
        set_trace_id(trace_id)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        # 流式响应在首个字节发出时就会返回,所以这里的 latency_ms 是首字节延迟,
        # 不是整段生成的总耗时。
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "event": "http.request",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "status_code": response.status_code,
            },
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("未处理异常")
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "trace_id": get_trace_id()},
        )

    app.include_router(auth.router)
    app.include_router(tickets.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)

    # 前端页面:纯静态资源,不走接口鉴权,数据全部由页面自己带 token 去取
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/docs", include_in_schema=False)
    async def docs():
        return render_docs_html(app)

    @app.get("/healthz", tags=["ops"], summary="健康检查")
    async def healthz() -> dict:
        store = build_vector_store(settings)
        embedder = build_embedder(settings)
        chunk_store = (
            build_chunk_vector_store(settings, embedder=embedder.name)
            if settings.chunking_enabled
            else None
        )
        reranker = build_reranker(settings)
        return {
            "status": "ok",
            "llm_enabled": settings.llm_enabled,
            "llm_model": settings.llm_model,
            "rate_limit_backend": getattr(app.state, "rate_limiter", None)
            and app.state.rate_limiter.backend_name,
            "retrieval_mode": settings.retrieval_mode,
            # 为 None 就说明向量检索没启用,知识库退化成纯关键词匹配
            "vector_backend": store.name if store else None,
            # 默认是离线哈希向量,配了 Embedding 之后这里会变成具体模型名
            "embedder": embedder.name,
            # 段落级索引:块大小与已索引的段落数,能一眼看出切块有没有生效
            "chunk_backend": chunk_store.name if chunk_store else None,
            "chunk_size": settings.knowledge_chunk_size if settings.chunking_enabled else None,
            "chunks_indexed": (
                await _count_indexed_chunks(embedder.name, embedder.dim)
                if chunk_store
                else None
            ),
            # 默认是零依赖的词法重排,配了重排 API 之后这里会变成具体模型名
            "reranker": reranker.name if reranker else None,
            "tools": app.state.agent.tool_names if hasattr(app.state, "agent") else [],
        }

    return app


app = create_app()
