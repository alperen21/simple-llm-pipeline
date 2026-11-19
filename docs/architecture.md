# Architecture

The installable package is `library_agent` under `src/`. Configuration is YAML
validated by Pydantic models with `extra="forbid"`, so unknown keys fail closed.

## Package map

```text
library_agent/
  schemas/          contracts: training, experiments, deployment, dataset, rag, catalog
  training/         CLI + SFT + sweep + selection + merge + Hub upload
  serving/          CLI + llama.cpp convert/quantize + Ollama
  rag/              Chroma store, ingest, grounded QA helper
  mcp/              MCPServer, Book schema, catalog + knowledge tools
  agent/            AgentState, system prompt, nodes, compiled graph
  cli/              reserved root Typer app (`library-agent`)
  domain/           planned entities, ports, services (stubs)
  infrastructure/   planned SQLite catalog (stubs)
  evaluation/       planned gates and benchmarks (stubs)
  observability/    planned tracing helpers (stubs)
```

## Runtime graph

The agent is a two-node LangGraph:

1. **agent** — ChatOpenAI with MCP tools bound; prepends `SYSTEM_PROMPT`.
2. **tools** — `ToolNode` executes requested tools and records names and counts
   on `AgentState`.

Edges: `START → agent → (tools | END)`. After tools, control returns to the
agent. A memory checkpointer keeps thread state for a `thread_id`.

MCP currently serves a **hard-coded catalog** (`mcp/tools.py`) plus live RAG
retrieval. Catalog persistence is intended to move behind `domain.ports` and
`infrastructure.catalog.sqlite`.

## Configuration files

| File | Schema | Used by |
| --- | --- | --- |
| `configs/training/pilot.yaml` | `TrainingConfig` | `library-train run`, sweep base |
| `configs/training/final.yaml` | `TrainingConfig` | production-sized SFT after selection |
| `configs/sweep.yaml` | `SweepConfig` | `library-train sweep create` |
| `configs/selection.yaml` | `SelectionConfig` | `library-train select run` |
| `configs/serving/deployment.yaml` | `DeploymentConfig` | `library-deploy *` |

RAG paths are code constants, not YAML: documents in `data/knowledge/`, Chroma
under `data/chroma/`, collection `library_knowledge`.

## Trust boundaries

- Retrieved passages and tool JSON are treated as untrusted data. The agent
  system prompt forbids following instructions inside documents.
- Deployment Hub repos default to private (`hf_public: false`).
- Secrets belong in `.env` (gitignored).
