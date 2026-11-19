"""Manage the persistent vector store and semantic retrieval."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

INDEX_DIR = Path("data/chroma")
COLLECTION_NAME = "library_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"


class RetrievedPassage(BaseModel):
    """A retrieved passage with its source and vector distance."""

    source: str
    start_index: int = Field(ge=0)
    content: str
    distance: float


def get_store() -> Chroma:
    """Open or create the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
        ),
        persist_directory=str(INDEX_DIR),
    )


def retrieve(query: str, k: int = 4) -> list[RetrievedPassage]:
    """Find the closest indexed passages to a natural-language query."""
    if not query.strip():
        raise ValueError("The query cannot be empty.")

    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10.")

    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "The vector index does not exist. Run python -m library_agent.rag.ingest first."
        )

    store = get_store()

    matches = store.similarity_search_with_score(
        query=query,
        k=k,
    )

    return [
        RetrievedPassage(
            source=str(document.metadata["source"]),
            start_index=int(document.metadata.get("start_index", 0)),
            content=document.page_content,
            distance=float(distance),
        )
        for document, distance in matches
    ]


def retrieve_context(
    query: str,
    k: int = 4,
) -> tuple[str, list[str]]:
    """Retrieve passages and format them as LLM context."""
    passages = retrieve(query=query, k=k)

    context = "\n\n".join(
        (
            f"[{index}] Source: {passage.source}\n"
            f"Start index: {passage.start_index}\n"
            f"{passage.content}"
        )
        for index, passage in enumerate(passages, start=1)
    )

    sources = list(dict.fromkeys(passage.source for passage in passages))

    return context, sources
