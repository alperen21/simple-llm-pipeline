# 0001 — Package boundaries

- Status: accepted
- Date: 2026-11-19

## Context

The repo mixes dataset prep, LoRA training, GGUF export, RAG, MCP, and a
LangGraph agent. A single CLI mixed training with deploy and made serving
depend on the training package import graph.

## Decision

- Put validated configs in `library_agent.schemas`.
- Keep SFT, sweeps, and selection under `library_agent.training` (`library-train`).
- Keep GGUF and Ollama under `library_agent.serving` (`library-deploy`).
- Keep retrieval in `library_agent.rag`, tools in `library_agent.mcp`, and the
  graph in `library_agent.agent`.
- Reserve `domain` / `infrastructure` for catalog persistence without changing
  MCP tool names.

## Consequences

Operators install the `training` extra only when they train or export. The
agent runtime depends on LangChain, MCP, and OpenAI embeddings, not TRL.
Merge still lives in `training.merge` because it needs PEFT; the deploy CLI
imports it lazily.
