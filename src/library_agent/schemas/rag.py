"""Schemas for the RAG store."""

from pydantic import BaseModel, Field


class RetrievedPassage(BaseModel):
    """A retrieved passage with its source and vector distance."""

    source: str
    start_index: int = Field(ge=0)
    content: str
    distance: float
