"""Create and run the library catalog MCP server."""

from mcp.server.mcpserver import MCPServer

from library_agent.mcp.tools import register_tools

mcp = MCPServer(
    "library-catalog",
    version="0.1.0",
    instructions=(
        "Use these tools to search the library catalog, "
        "retrieve book information, and check availability."
    ),
)

register_tools(mcp)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        stateless_http=True,
        json_response=True,
    )
