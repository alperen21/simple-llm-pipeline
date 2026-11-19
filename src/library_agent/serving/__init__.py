"""Package a LoRA adapter for local inference.

``cli`` is the ``library-deploy`` Typer app. ``export`` converts merged
Hugging Face weights to GGUF and quantizes them with llama.cpp. ``ollama``
registers the quantized file with a local Ollama server.
"""
