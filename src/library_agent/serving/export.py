"""Convert Transformers checkpoints to GGUF and quantize them."""

import subprocess
from pathlib import Path

from library_agent.schemas.deployment import DeploymentConfig


def convert_to_gguf(config: DeploymentConfig) -> Path:
    """Convert a merged Transformers model to FP16 GGUF."""
    model_dir = config.merged_dir.resolve()
    output_dir = config.gguf_dir.resolve()

    converter = config.llama_cpp_dir.resolve() / "convert_hf_to_gguf.py"

    python = config.conversion_python.resolve()

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found: {model_dir}")

    if not converter.is_file():
        raise FileNotFoundError(converter)

    if not python.is_file():
        raise FileNotFoundError(python)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model-f16.gguf"

    command = [
        str(python),
        str(converter),
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        "f16",
    ]

    subprocess.run(command, check=True)

    if not output_path.is_file():
        raise RuntimeError("GGUF conversion produced no output.")

    return output_path


def find_quantizer(llama_cpp_dir: Path) -> Path:
    """Locate the compiled llama.cpp quantization tool."""
    root = llama_cpp_dir.resolve()

    candidates = [
        root / "build/bin/llama-quantize",
        root / "build/bin/Release/llama-quantize",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("llama-quantize not found. Build llama.cpp first.")


def quantize_gguf(config: DeploymentConfig) -> Path:
    """Quantize an FP16 GGUF model.

    Returns:
        Path to the quantized GGUF artifact.
    """
    gguf_dir = config.gguf_dir.resolve()

    source = gguf_dir / "model-f16.gguf"
    destination = gguf_dir / f"model-{config.quantization}.gguf"

    if not source.is_file():
        raise FileNotFoundError(source)

    quantizer = find_quantizer(config.llama_cpp_dir)

    subprocess.run(
        [
            str(quantizer),
            str(source),
            str(destination),
            config.quantization,
        ],
        check=True,
    )

    if not destination.is_file():
        raise RuntimeError("Quantization completed without an output file.")

    return destination
