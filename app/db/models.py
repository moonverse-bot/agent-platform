"""数据模型:用户、工单、会话、消息、工具调用轨迹、知识库文章与段落。"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(64), default="")
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="open")
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    owner: Mapped[User] = relationship(back_populates="tickets")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ToolCallLog(Base):
    """每次工具调用的轨迹,是后面做可观测和评测的数据来源。"""

    __tablename__ = "tool_call_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    arguments: Mapped[str] = mapped_column(Text, default="")
    result_preview: Mapped[str] = mapped_column(Text, default="")
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeArticle(Base):
    __tablename__ = "knowledge_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    tags: Mapped[str] = mapped_column(String(200), default="")


class KnowledgeVector(Base):
    """知识库向量索引(SQLite / 便携后端)。

    向量以 JSON 文本存储,检索时在进程内算余弦。Postgres 上不走这张表,
    而是由 `app/agent/retrieval.py` 里的 PgVectorStore 建原生 `vector(N)` 列,
    这样两种方言各自用最合适的存储,不需要在 ORM 里写方言相关的类型。

    `embedder` 与 `dim` 一起构成索引的「版本」:换了 Embedding 模型或维度,
    旧向量会被视为缺失并重新计算。
    """

    __tablename__ = "knowledge_vectors"

    article_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_articles.id", ondelete="CASCADE"), primary_key=True
    )
    embedder: Mapped[str] = mapped_column(String(64), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class KnowledgeChunk(Base):
    """知识库文档的段落(切块)与段落级向量。

    一篇文档按 `app/agent/chunking.py` 的规则切成若干块,每块单独出一个向量:
    整篇一个向量会把标题、标签、正文平均成「一个点」,文档越长,单句信息越容易被稀释;
    切成段落后,只要有一小段正好在讲用户问的事,那一小段就能精确命中。

    这张表同时是段落正文的唯一来源(两种数据库都用它读正文),而向量列只在
    SQLite 侧使用:Postgres 上向量存在 `PgChunkVectorStore` 的原生
    `knowledge_chunk_embeddings` 表里,和文章级的 knowledge_vectors /
    knowledge_embeddings 是同一套分工。

    (article_id, ordinal) 唯一:同一篇文档里的块序号不会重复;
    `embedder` / `dim` 与文章级一样构成索引的「版本」,换模型或换维度会自动重算。
    """

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("article_id", "ordinal", name="uq_knowledge_chunks_article_ordinal"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_articles.id", ondelete="CASCADE"), index=True
    )
    #: 块在文档里的序号,从 0 开始;按它排序就能还原原文顺序
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedder: Mapped[str] = mapped_column(String(64), default="", index=True)
    dim: Mapped[int] = mapped_column(Integer, default=0)
    vector: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
