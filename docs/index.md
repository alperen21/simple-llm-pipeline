# Library Agent documentation

This project is a vertical slice of a production LLM system: public tool-calling
data in, a locally served library assistant out.

## Pipeline

```text
Hugging Face dataset
        │
        ▼
 library-train prepare     validated JSONL + SHA-256 manifest
        │
        ▼
 library-train sweep       W&B LoRA hyperparameter trials
        │
        ▼
 library-train select      cost and quality constraints
        │
        ▼
 library-train run         PEFT adapter under artifacts/
        │
        ▼
 library-deploy merge      full Transformers weights
        │
        ▼
 library-deploy convert    FP16 GGUF via llama.cpp
        │
        ▼
 library-deploy quantize   Q4_K_M / Q5_K_M / Q8_0
        │
        ▼
 library-deploy ollama     local Ollama model tag
```

In parallel, library policy Markdown is indexed into Chroma. The MCP server
exposes catalog search plus `search_knowledge`. LangGraph binds those tools to
an LLM and traces runs in LangSmith when configured.

## Read next

1. [Architecture](architecture.md) — package boundaries and data flow
2. [Training](training.md) — prepare, sweep, select, fine-tune
3. [Deployment](deployment.md) — merge, GGUF, Ollama, Hub
4. [Agent and RAG](agent.md) — ingest, MCP, LangGraph
5. [API reference](reference.md) — generated from Google-style docstrings
