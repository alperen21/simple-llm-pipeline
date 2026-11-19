"""Test RAG retrieval through the MCP server."""

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    """Test MCP tool discovery and semantic retrieval."""
    async with (
        streamable_http_client("http://127.0.0.1:8000/mcp") as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Discover registered MCP tools.
        response = await session.list_tools()

        print("Available tools:")
        for tool in response.tools:
            print(f"- {tool.name}: {tool.description}")

        # Verify that the RAG tool is registered.
        assert any(tool.name == "search_knowledge" for tool in response.tools), (
            "RAG tool is not registered!"
        )

        # Retrieve passages using semantic search.
        result = await session.call_tool(
            "search_knowledge",
            arguments={
                "query": "How long can library members borrow books?",
                "limit": 3,
            },
        )

        if result.is_error:
            raise RuntimeError(f"RAG retrieval failed: {result.content}")

        print("\nRetrieved passages:")
        if result.structured_content is not None:
            print(result.structured_content)
        else:
            for content in result.content:
                print(content)

        print("\nRAG retrieval completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
