"""Ingest local Markdown and text documents into Chroma."""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from library_agent.rag.store import get_store

DOCUMENT_DIR = Path("data/knowledge")


def ingest_documents(directory: Path = DOCUMENT_DIR) -> int:
    """Index documents and replace previously indexed versions."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge directory does not exist: {directory}")

    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )

    if not paths:
        raise ValueError("No Markdown or text documents found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )

    store = get_store()
    total_chunks = 0

    for path in paths:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        source = path.relative_to(directory).as_posix()

        document = Document(
            page_content=content,
            metadata={"source": source},
        )

        chunks = splitter.split_documents([document])

        # Remove old chunks for this source before re-indexing.
        existing = store.get(
            where={"source": source},
            include=[],
        )
        existing_ids = existing["ids"]

        if existing_ids:
            store.delete(ids=existing_ids)

        # Stable IDs make each chunk identifiable.
        ids = [
            hashlib.sha256(f"{source}:{index}".encode()).hexdigest() for index in range(len(chunks))
        ]

        store.add_documents(
            documents=chunks,
            ids=ids,
        )

        total_chunks += len(chunks)

    return total_chunks


def main() -> None:
    """Run document ingestion from the command line."""
    count = ingest_documents()
    print(f"Indexed {count} document chunks.")


if __name__ == "__main__":
    main()
