"""Library catalog assistant: train, package, retrieve, and serve.

The package is a pipeline, not a single runtime. Use ``library-train`` for
data and LoRA, ``library-deploy`` for GGUF/Ollama, ``library_agent.rag`` for
the knowledge index, ``library_agent.mcp`` for tools, and
``library_agent.agent`` for the LangGraph loop.
"""
