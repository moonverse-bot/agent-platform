"""内置工具集。"""

from app.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from app.agent.tools.knowledge import KNOWLEDGE_TOOLS
from app.agent.tools.tickets import TICKET_TOOLS

__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
]


def build_default_registry() -> ToolRegistry:
    return ToolRegistry([*TICKET_TOOLS, *KNOWLEDGE_TOOLS])
