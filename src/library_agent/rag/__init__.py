"""Retrieval-augmented generation over local library knowledge.

Ingest Markdown/text from ``data/knowledge/`` into a persistent Chroma
collection, retrieve passages with OpenAI embeddings, and optionally answer
questions with a grounded Ollama chat model. MCP exposes the same retrieve
path as ``search_knowledge``.
"""
