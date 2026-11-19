"""Define retrieval, LLM, and MCP tool-execution nodes."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from library_agent.agent.prompts import (
    SYSTEM_PROMPT,
    augment_user_question,
)
from library_agent.agent.state import AgentState
from library_agent.rag.store import retrieve_context

NodeFunction = Callable[
    [AgentState],
    Awaitable[dict[str, Any]],
]


def latest_user_message(
    messages: Sequence[BaseMessage],
) -> HumanMessage:
    """Find the most recent user message in the conversation."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message

    raise ValueError("No user message found in the conversation.")


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve RAG context for the latest user question."""
    message = latest_user_message(state["messages"])

    if not isinstance(message.content, str):
        raise ValueError("RAG currently requires a text question.")

    # Run synchronous retrieval without blocking the event loop.
    context, sources = await asyncio.to_thread(
        retrieve_context,
        message.content,
        4,
    )

    return {
        "rag_context": context,
        "rag_sources": sources,
    }


def make_agent_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
) -> NodeFunction:
    """Create an agent node with automatically injected RAG context."""
    llm_with_tools = llm.bind_tools(list(tools))

    async def agent_node(state: AgentState) -> dict[str, Any]:
        """Invoke the LLM using retrieved context and conversation history."""
        messages = list(state["messages"])

        # Augment only the latest user message.
        # Do not modify the message stored in LangGraph state.
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]

            if isinstance(message, HumanMessage):
                if not isinstance(message.content, str):
                    raise ValueError("Expected a text user message.")

                messages[index] = message.model_copy(
                    update={
                        "content": augment_user_question(
                            question=message.content,
                            rag_context=state.get("rag_context", ""),
                        )
                    }
                )
                break
        else:
            raise ValueError("No user message found.")

        response = await llm_with_tools.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *messages,
            ]
        )

        return {"messages": [response]}

    return agent_node


def make_tool_node(
    tools: Sequence[BaseTool],
) -> NodeFunction:
    """Create a node that executes MCP tools and tracks usage."""
    executor = ToolNode(list(tools))

    async def execute_tools(state: AgentState) -> dict[str, Any]:
        """Execute the tools requested by the agent."""
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            raise ValueError("Expected an AIMessage before tool execution.")

        tool_names = [call["name"] for call in last_message.tool_calls]

        result = await executor.ainvoke(state)

        if not isinstance(result, dict):
            raise TypeError("Unexpected ToolNode result.")

        return {
            "messages": result["messages"],
            "tool_call_count": len(tool_names),
            "tools_used": tool_names,
        }

    return execute_tools
