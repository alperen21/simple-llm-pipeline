"""Tools for the MCP server."""

from mcp.server.mcpserver import MCPServer

from library_agent.mcp.schemas import Book

# Temporary catalog for testing MCP integration.
# Later, these operations will call our catalog service.
BOOKS = (
    Book(
        book_id="book-001",
        title="Frankenstein",
        author="Mary Shelley",
        available_copies=3,
    ),
    Book(
        book_id="book-002",
        title="Pride and Prejudice",
        author="Jane Austen",
        available_copies=2,
    ),
    Book(
        book_id="book-003",
        title="Dracula",
        author="Bram Stoker",
        available_copies=0,
    ),
)


def find_books(query: str, limit: int = 10) -> list[Book]:
    """Search books by title or author."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= limit <= 50:
        raise ValueError("Limit must be between 1 and 50.")

    normalized_query = query.strip().casefold()

    return [
        book
        for book in BOOKS
        if (normalized_query in book.title.casefold() or normalized_query in book.author.casefold())
    ][:limit]


def find_book_by_id(book_id: str) -> Book:
    """Retrieve a book by its unique identifier."""
    for book in BOOKS:
        if book.book_id == book_id:
            return book

    raise ValueError(f"Book '{book_id}' was not found.")


def register_tools(mcp: MCPServer) -> None:
    """Register library catalog tools with an MCP server."""

    @mcp.tool()
    def search_books(query: str, limit: int = 10) -> list[Book]:
        """Search the library catalog by book title or author."""
        return find_books(query=query, limit=limit)

    @mcp.tool()
    def get_book(book_id: str) -> Book:
        """Retrieve a book's details using its book ID."""
        return find_book_by_id(book_id)

    @mcp.tool()
    def check_availability(book_id: str) -> bool:
        """Check whether a book currently has an available copy."""
        book = find_book_by_id(book_id)
        return book.available_copies > 0
