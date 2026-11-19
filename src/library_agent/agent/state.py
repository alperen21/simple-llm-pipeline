"""Define the shared state for the library agent."""

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation state, retrieved context, and tool usage."""

    rag_context: str
    rag_sources: list[str]

    tool_call_count: Annotated[int, operator.add]
    tools_used: Annotated[list[str], operator.add]
