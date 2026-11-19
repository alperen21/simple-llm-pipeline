"""Command-line interface for model packaging and deployment."""

from pathlib import Path
from typing import Any

import typer
import yaml

from library_agent.schemas.deployment import DeploymentConfig

app = typer.Typer(help="Package and deploy a trained adapter.")

DEFAULT_CONFIG = Path("configs/serving/deployment.yaml")


def load_config(path: Path) -> DeploymentConfig:
    """Load and validate a deployment YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Deployment configuration must be a YAML mapping.")

    return DeploymentConfig.model_validate(raw)


@app.command()
def merge(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
) -> None:
    """Merge the LoRA adapter into the base model."""
    from library_agent.training.merge import merge_adapter

    deployment = load_config(config)
    output_dir = merge_adapter(deployment)

    typer.echo(f"Merged model saved to: {output_dir}")


@app.command()
def convert(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
) -> None:
    """Convert the merged Transformers model to FP16 GGUF."""
    from library_agent.serving.export import convert_to_gguf

    deployment = load_config(config)
    output_path = convert_to_gguf(deployment)

    typer.echo(f"GGUF saved to: {output_path}")


@app.command()
def quantize(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
) -> None:
    """Quantize the FP16 GGUF artifact."""
    from library_agent.serving.export import quantize_gguf

    deployment = load_config(config)
    output_path = quantize_gguf(deployment)

    typer.echo(f"Quantized GGUF saved to: {output_path}")


@app.command()
def ollama(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
) -> None:
    """Register the quantized GGUF with a local Ollama server."""
    from library_agent.serving.ollama import create_ollama_model

    deployment = load_config(config)
    model_name = create_ollama_model(deployment)

    typer.echo(f"Ollama model created: {model_name}")


@app.command("publish-adapter")
def publish_adapter_cmd(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
    model_card: Path = typer.Option(
        ...,
        "--model-card",
        help="Path to the adapter model card.",
    ),
) -> None:
    """Publish the LoRA adapter to the Hugging Face Hub."""
    from library_agent.training.publish import publish_adapter

    deployment = load_config(config)
    url = publish_adapter(deployment, model_card)

    typer.echo(f"Adapter published: {url}")


@app.command("publish-gguf")
def publish_gguf_cmd(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
    model_card: Path = typer.Option(
        ...,
        "--model-card",
        help="Path to the GGUF model card.",
    ),
    license_file: Path = typer.Option(
        ...,
        "--license",
        help="Path to LICENSE.txt.",
    ),
    notice_file: Path = typer.Option(
        ...,
        "--notice",
        help="Path to NOTICE.",
    ),
) -> None:
    """Publish the quantized GGUF to the Hugging Face Hub."""
    from library_agent.training.publish import publish_gguf

    deployment = load_config(config)
    url = publish_gguf(
        deployment,
        model_card,
        license_file,
        notice_file,
    )

    typer.echo(f"GGUF published: {url}")


@app.command()
def run(
    config: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="Path to the deployment YAML configuration.",
    ),
) -> None:
    """Merge, convert, quantize, and register with Ollama."""
    from library_agent.serving.export import (
        convert_to_gguf,
        quantize_gguf,
    )
    from library_agent.serving.ollama import create_ollama_model
    from library_agent.training.merge import merge_adapter

    deployment = load_config(config)

    merged_dir = merge_adapter(deployment)
    typer.echo(f"Merged model saved to: {merged_dir}")

    gguf_path = convert_to_gguf(deployment)
    typer.echo(f"GGUF saved to: {gguf_path}")

    quantized_path = quantize_gguf(deployment)
    typer.echo(f"Quantized GGUF saved to: {quantized_path}")

    model_name = create_ollama_model(deployment)
    typer.echo(f"Ollama model created: {model_name}")


if __name__ == "__main__":
    app()
