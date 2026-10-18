"""工单相关的出入参模型。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TicketStatus = Literal["open", "processing", "closed"]
TicketPriority = Literal["low", "normal", "high", "urgent"]


class TicketCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    priority: TicketPriority = "normal"


class TicketUpdate(BaseModel):
    status: TicketStatus | None = None
    priority: TicketPriority | None = None


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime
