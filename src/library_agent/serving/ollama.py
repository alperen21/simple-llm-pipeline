"""Register GGUF model artifacts with a local Ollama server."""

import shutil
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from library_agent.schemas import DeploymentConfig


def check_ollama() -> None:
    """Check the Ollama executable and local API availability."""
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama executable not found in PATH.")

    try:
        with urlopen(
            "http://127.0.0.1:11434/api/tags",
            timeout=5,
        ) as response:
            if response.status != 200:
                raise RuntimeError("Ollama API returned an unexpected status.")

    except (OSError, URLError) as exc:
        raise RuntimeError("Ollama is not running. Start the local server.") from exc


def create_modelfile(config: DeploymentConfig) -> Path:
    """Generate an Ollama Modelfile for the quantized GGUF."""
    gguf = config.gguf_dir.resolve() / f"model-{config.quantization}.gguf"

    if not gguf.is_file():
        raise FileNotFoundError(gguf)

    modelfile = gguf.parent / "Modelfile"

    content = (
        f"FROM ./{gguf.name}\nPARAMETER temperature 0\nPARAMETER num_ctx {config.context_length}\n"
    )

    modelfile.write_text(
        content,
        encoding="utf-8",
    )

    return modelfile


def create_ollama_model(config: DeploymentConfig) -> str:
    """Register the quantized model with Ollama."""
    check_ollama()

    modelfile = create_modelfile(config)

    subprocess.run(
        [
            "ollama",
            "create",
            config.ollama_model,
            "-f",
            str(modelfile),
        ],
        cwd=modelfile.parent,
        check=True,
    )

    return config.ollama_model
