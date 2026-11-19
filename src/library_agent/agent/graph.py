"""Build and run the automatically RAG-augmented library agent."""

import asyncio
import os
from collections.abc import Sequence

from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from library_agent.agent.nodes import (
    make_agent_node,
    make_tool_node,
    retrieve_node,
)
from library_agent.agent.state import AgentState

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MODEL = "gpt-4.1-mini"


def build_graph(
    tools: Sequence[BaseTool],
    model_name: str = DEFAULT_MODEL,
) -> CompiledStateGraph:
    """Build the RAG-first ReAct agent."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve_node)

    builder.add_node(
        "agent",
        make_agent_node(llm, tools),
    )

    builder.add_node(
        "tools",
        make_tool_node(tools),
    )

    # Automatic RAG on every incoming user message.
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "agent")

    # ReAct tool-calling loop.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    builder.add_edge("tools", "agent")

    return builder.compile(
        checkpointer=InMemorySaver(),
    )


async def main() -> None:
    """Connect to MCP and run the RAG-augmented agent."""
    mcp_url = os.getenv("LIBRARY_MCP_URL", DEFAULT_MCP_URL)

    async with MCPAdapter(mcp_url) as adapter:
        tools = await adapter.list_tools()

        print("Discovered MCP tools:")
        for tool in tools:
            print(f"- {tool.name}")

        graph = build_graph(tools)

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Find the book Frankenstein in the library catalog "
                            "and check whether it is currently available."
                        )
                    )
                ],
            },
            config={
                "configurable": {
                    "thread_id": "library-demo",
                },
                "run_name": "library-agent",
                "tags": ["library", "rag", "mcp"],
                "recursion_limit": 20,
            },
        )

        print("\nFinal answer:")
        print(result["messages"][-1].content)

        print("\nRAG sources:")
        for source in result.get("rag_sources", []):
            print(f"- {source}")

        print("\nMCP tool usage:")
        print("Tool calls:", result.get("tool_call_count", 0))
        print("Tools used:", result.get("tools_used", []))


if __name__ == "__main__":
    asyncio.run(main())
