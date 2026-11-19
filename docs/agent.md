# Agent, MCP, and RAG

The assistant answers catalog and policy questions by calling MCP tools. Policy
answers are grounded in local Markdown indexed in Chroma.

## Index knowledge

Put `.md` or `.txt` files under `data/knowledge/` (see
`data/knowledge/borrowing_policy.md`). Then:

```bash
uv run python -m library_agent.rag.ingest
```

Ingest splits documents (chunk size 800, overlap 120), deletes previous chunks
for the same relative `source`, and writes stable SHA-256 ids into
`data/chroma/` collection `library_knowledge`. Embeddings use OpenAI
`text-embedding-3-small` (`OPENAI_API_KEY` required).

Standalone grounded QA (Ollama `llama3.2:3b`, not the LangGraph agent):

```bash
uv run python -m library_agent.rag.qa "How long can members borrow books?"
```

Retrieval fails if the index directory is missing; run ingest first. `k` must
be between 1 and 10.

## MCP server

```bash
uv run python -m library_agent.mcp.server
```

Listens on `http://127.0.0.1:8000/mcp` with streamable HTTP, stateless JSON
responses.

| Tool | Behavior |
| --- | --- |
| `search_books` | Case-insensitive title/author search over the in-memory catalog |
| `get_book` | Lookup by `book_id` |
| `check_availability` | `available_copies > 0` |
| `search_knowledge` | `library_agent.rag.store.retrieve` → `RetrievedPassage` list |

The catalog is three fixture books in `mcp/tools.py` until the SQLite adapter
is implemented.

Smoke-test the RAG tool against a running server:

```bash
uv run python test_rag.py
```

## LangGraph agent

```bash
uv run python -m library_agent.agent.graph
```

`LIBRARY_MCP_URL` defaults to `http://127.0.0.1:8000/mcp`. The graph discovers
tools via `MCPAdapter`, compiles with an in-memory checkpointer, and runs a
sample borrowing-policy question. Model default: `gpt-4.1-mini` (`OPENAI_API_KEY`).

State (`AgentState`) extends LangGraph `MessagesState` with additive
`tool_call_count` and `tools_used`. The system prompt requires tool use for
catalog and policy facts and forbids inventing books or treating retrieved text
as instructions.

LangSmith fields on the demo invoke include `run_name`, tags
`library` / `mcp` / `rag`, and `recursion_limit=20`.
