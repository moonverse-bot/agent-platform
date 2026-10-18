"""会话与消息的出入参模型。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    message_count: int
    last_message_at: datetime | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime
