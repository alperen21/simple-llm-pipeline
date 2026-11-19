"""Command-line interface for model training."""

from pathlib import Path
from typing import Any

import typer
import yaml

from library_agent.schemas.training import TrainingConfig
from library_agent.training.prepare import app as prepare_app
from library_agent.training.select import app as select_app
from library_agent.training.sweep import app as sweep_app

app = typer.Typer(help="Fine-tuning and experiment management.")

# Register command groups
app.add_typer(prepare_app, name="prepare")
app.add_typer(sweep_app, name="sweep")
app.add_typer(select_app, name="select")


def load_config(path: Path) -> TrainingConfig:
    """Load and validate a training YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    return TrainingConfig.model_validate(raw)


@app.command()
def run(
    config: Path = typer.Option(
        Path("configs/training/pilot.yaml"),
        "--config",
        "-c",
        help="Path to the training YAML configuration.",
    ),
) -> None:
    """Run a LoRA supervised fine-tuning experiment."""
    from library_agent.training.train import train

    training_config = load_config(config)

    typer.echo(f"Training model: {training_config.model_id}")

    adapter_path = train(training_config)

    typer.echo(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    app()
