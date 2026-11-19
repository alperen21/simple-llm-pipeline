# Library Agent

Production-style pipeline for preparing tool-calling data, LoRA fine-tuning a
small Llama model, packaging it for local inference, and serving a library
assistant over MCP with retrieval-augmented policy answers.

## What this repository contains

| Layer | Package | Responsibility |
| --- | --- | --- |
| Schemas | `library_agent.schemas` | Pydantic configs for training, sweeps, selection, deployment, RAG |
| Training | `library_agent.training` | Dataset prep, SFT, W&B sweeps, cost-aware selection, merge, Hub publish |
| Serving | `library_agent.serving` | GGUF conversion, quantization, Ollama registration |
| RAG | `library_agent.rag` | Markdown ingest into Chroma, semantic retrieval, grounded QA |
| MCP | `library_agent.mcp` | Catalog and knowledge tools over streamable HTTP |
| Agent | `library_agent.agent` | LangGraph ReAct loop bound to MCP tools |

Placeholder packages (`domain`, `evaluation`, `infrastructure`, `observability`)
are reserved for the next layers. Do not treat them as implemented APIs yet.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- For training and GGUF export: the `training` extra, a GPU-capable PyTorch
  wheel for your platform, and a local [llama.cpp](https://github.com/ggerganov/llama.cpp) build
- For the live agent: an OpenAI API key (embeddings + ChatOpenAI), Ollama with
  `llama3.2:3b` if you use the standalone RAG QA script, and the MCP server

## Install

```bash
uv sync --group dev
uv sync --group training   # LoRA, TRL, datasets, W&B
```

Copy `.env.example` to `.env` when you add secrets. Never commit `.env`.

## Command-line entry points

```bash
uv run --group training library-train --help
uv run --group training library-deploy --help
```

| Command | Module | Purpose |
| --- | --- | --- |
| `library-train` | `library_agent.training.cli` | `prepare`, `sweep`, `select`, `run` |
| `library-deploy` | `library_agent.serving.cli` | `merge`, `convert`, `quantize`, `ollama`, publish |

Runtime pieces are modules, not the training CLI:

```bash
uv run python -m library_agent.rag.ingest
uv run python -m library_agent.mcp.server
uv run python -m library_agent.agent.graph
```

Full walkthroughs live in [`docs/`](docs/index.md). Build the site with:

```bash
uv run --group dev mkdocs serve
```
