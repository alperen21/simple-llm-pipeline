"""Model Context Protocol server for catalog and knowledge tools.

``server`` constructs ``MCPServer("library-catalog")`` and runs streamable
HTTP on ``127.0.0.1:8000``. ``tools`` registers search, lookup, availability,
and RAG retrieval. ``schemas.Book`` is the catalog record returned to clients.
"""
