# Development

## Tooling

| Tool | Config | Role |
| --- | --- | --- |
| uv | `pyproject.toml` | Env, extras, scripts |
| Ruff | `[tool.ruff]` | Lint + format, Google docstrings, Python 3.12 |
| Mypy | `[tool.mypy]` strict + Pydantic plugin | Types |
| pytest | `[tool.pytest.ini_options]` | Tests |
| pre-commit | `.pre-commit-config.yaml` | `ruff check --fix` then `ruff format` |
| MkDocs | `mkdocs.yml` | This site |

```bash
uv sync --group dev
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest
uv run mkdocs serve
```

Ruff treats `typer.Option` and `typer.Argument` as immutable calls (`B008`).
Docstrings use the Google convention (`D`). Public modules and packages need
module-level docstrings (`D100` / `D104`).

## Layout conventions

- New validated YAML belongs in `configs/` and a Pydantic model in `schemas/`.
- Training-only imports stay inside command bodies (`from library_agent.training.train import train`) so `library-train --help` can load without GPU stacks where possible. `prepare` still imports `huggingface_hub` at module level.
- Serving CLI lazy-imports merge, export, Ollama, and publish for the same reason.
- Do not nest deploy commands under `library-train`.

## Environment

Typical variables:

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | RAG embeddings, LangGraph ChatOpenAI |
| `LIBRARY_MCP_URL` | Agent MCP client (default `http://127.0.0.1:8000/mcp`) |
| `LANGSMITH_*` / `LANGCHAIN_*` | Optional tracing |
| Hugging Face token | dataset download, Hub publish |
| `WANDB_API_KEY` | sweeps |

`.env` is gitignored. `data/chroma/` is local index state; `artifacts/` is
training output.

## Architecture Decision Records

Add dated markdown under `docs/adr/` when you change a boundary (for example
catalog storage or embedding provider). Keep each ADR short: context, decision,
consequences.
