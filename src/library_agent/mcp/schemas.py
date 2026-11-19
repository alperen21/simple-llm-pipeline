"""Schemas for the MCP server."""

from pydantic import BaseModel, Field


class Book(BaseModel):
    """A book in the library catalog."""

    book_id: str
    title: str
    author: str
    available_copies: int = Field(ge=0)
