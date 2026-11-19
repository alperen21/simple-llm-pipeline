"""Test the MCP server."""

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    """Test the MCP server."""
    async with (
        streamable_http_client("http://127.0.0.1:8000/mcp") as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Discover registered MCP tools.
        response = await session.list_tools()

        for tool in response.tools:
            print(tool.name, tool.description)

        # Invoke an MCP tool.
        result = await session.call_tool(
            "search_books",
            arguments={"query": "Frankenstein"},
        )

        print(result)


asyncio.run(main())
