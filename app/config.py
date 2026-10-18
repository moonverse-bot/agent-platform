"""应用配置:全部来自环境变量或 .env,代码里不出现任何密钥。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = "agent-platform"
    debug: bool = False

    # 首次启动时是否创建演示账号 demo@example.com,方便前端预填的登录信息开箱即用。
    # 线上部署务必设为 false,否则任何人都能用一个已知密码登录演示账号。
    seed_demo_data: bool = True

    # 数据层
    database_url: str = "sqlite+aiosqlite:///./dev.db"
    redis_url: str = ""

    # 鉴权
    # HS256 要求密钥足够长,低于 32 字节 PyJWT 会告警,这里给一个合规的本地默认值
    jwt_secret: str = "dev-only-secret-change-me-before-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 720

    # 大模型:沿用 DeepSeek 官方环境变量名,便于和已有项目共用
    llm_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(
        default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL"
    )
    llm_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 60.0
    llm_max_tool_rounds: int = 4

    # 知识库检索
    # mode: hybrid(关键词 + 向量排名融合,默认)| keyword(只走关键词)| vector(只走向量)
    retrieval_mode: str = "hybrid"
    retrieval_top_k: int = 3
    # 两路召回各取多少候选参与融合,最后再截到 top_k
    retrieval_candidates: int = 20
    # 向量最低相似度,低于它就不算命中。留 0 表示用当前 embedder 自带的建议阈值
    # (离线哈希向量约 0.12,神经 embedding 约 0.3 —— 两者分数量纲差别很大)。
    retrieval_min_vector_score: float = 0.0

    # 文档切块:长文档先切成段,每段单独出一个向量。
    # 块越大上下文越完整,但一个向量里混的主题越多、精度越低;
    # 300 字大约是「一个完整操作步骤」的体量,60 字重叠是为了别让答案正好掉在切口上。
    # 设为 0 表示关闭切块,知识库只保留「整篇一个向量」。
    knowledge_chunk_size: int = 300
    knowledge_chunk_overlap: int = 60

    # 重排:召回之后再把候选按与问题的相关度重排一遍(见 app/agent/rerank.py)
    # provider: lexical(零依赖的词法覆盖度,默认)| api(OpenAI 兼容的 /rerank 服务)
    rerank_enabled: bool = True
    rerank_provider: str = "lexical"
    rerank_api_key: str = Field(default="", validation_alias="RERANK_API_KEY")
    rerank_base_url: str = Field(default="", validation_alias="RERANK_BASE_URL")
    rerank_model: str = Field(default="", validation_alias="RERANK_MODEL")
    rerank_timeout_seconds: float = 10.0
    # 送进重排的候选条数:重排比召回贵,只重排头部这一段最划算
    rerank_candidate_limit: int = 10

    # 向量后端:auto 按数据库方言自动选(pgvector / sqlite),none 表示关掉向量检索
    vector_backend: str = "auto"

    # Embedding(OpenAI 兼容的 /embeddings 协议:OpenAI、硅基流动、智谱、DashScope 都兼容)
    # auto: 同时配了 key 与 model 就走在线服务,否则用离线哈希兜底
    embedding_provider: str = "auto"
    embedding_api_key: str = Field(default="", validation_alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="", validation_alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(default="", validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = 1024
    # 这条链路在对话里是同步等待的,超时要短:挂掉的服务不能把每次提问都拖住 30 秒
    embedding_timeout_seconds: float = 10.0

    # 限流
    rate_limit_per_minute: int = 20

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")

    @property
    def chunking_enabled(self) -> bool:
        """是否给长文档切块建段落级索引(chunk_size <= 0 表示关闭)。"""
        return self.knowledge_chunk_size > 0

    @property
    def keyword_search_enabled(self) -> bool:
        """关键词一路是否参与召回:hybrid / keyword 走它,纯向量模式不走。"""
        return self.retrieval_mode.strip().lower() in {"keyword", "hybrid"}

    @property
    def vector_search_enabled(self) -> bool:
        """向量一路是否参与召回。"""
        return self.retrieval_mode.strip().lower() in {"vector", "hybrid"}

    @property
    def online_embeddings_enabled(self) -> bool:
        """在线 Embedding 是否可用:显式指定 openai,或 auto 模式下 key 与 model 都齐了。"""
        provider = self.embedding_provider.strip().lower()
        if provider == "openai":
            return True
        return bool(
            provider == "auto"
            and self.embedding_api_key.strip()
            and self.embedding_model.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
