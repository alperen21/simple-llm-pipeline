"""LoRA hyperparameter optimization with W&B Sweeps."""

import json
from functools import partial
from pathlib import Path

import typer
import wandb
import yaml

from library_agent.schemas.experiments import SweepConfig
from library_agent.schemas.training import TrainingConfig

app = typer.Typer(help="Manage LoRA hyperparameter sweeps.")


def load_sweep_config(path: Path) -> SweepConfig:
    """Load and validate the W&B sweep configuration."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return SweepConfig.model_validate(data)


def load_training_config(path: Path) -> TrainingConfig:
    """Load the fixed pilot training configuration."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return TrainingConfig.model_validate(data)


def build_trial_config(
    base: TrainingConfig,
    parameters: dict[str, int | float],
    run_id: str,
    project: str,
) -> TrainingConfig:
    """Build a fully validated configuration for one trial."""
    allowed = {
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
    }

    unknown = set(parameters) - allowed

    if unknown:
        raise ValueError(f"Unknown trial parameters: {sorted(unknown)}")

    data = base.model_dump(mode="python")

    data.update(parameters)

    # Each trial must have its own checkpoint directory.
    data["output_dir"] = base.output_dir / run_id

    # Enable W&B tracking inside the TRL trainer.
    data["wandb_enabled"] = True
    data["wandb_project"] = project

    return TrainingConfig.model_validate(data)


def run_trial(
    base_config_path: Path,
    project: str,
) -> None:
    """Train and evaluate one W&B-provided configuration."""
    from library_agent.training.train import train

    base = load_training_config(base_config_path)

    with wandb.init(
        project=project,
        job_type="pilot-lora",
    ) as run:
        if run is None:
            raise RuntimeError("Failed to initialize W&B run.")

        allowed = (
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
        )

        parameters = {key: run.config[key] for key in allowed if key in run.config}

        config = build_trial_config(
            base=base,
            parameters=parameters,
            run_id=run.id,
            project=project,
        )

        run.config.update(
            {
                "model_id": config.model_id,
                "seed": config.seed,
                "max_length": config.max_length,
                "train_path": str(config.train_path),
                "validation_path": str(config.validation_path),
            },
            allow_val_change=True,
        )

        # Reuses the existing SFTTrainer implementation.
        adapter_path = train(config)

        # train.py must write this file after evaluation.
        metrics_path = config.output_dir / "metrics.json"

        if not metrics_path.is_file():
            raise FileNotFoundError(f"Training metrics missing: {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        eval_loss = float(metrics["eval_loss"])

        # Explicit final sweep objective.
        run.log({"selection/eval_loss": eval_loss})

        run.summary["adapter_path"] = str(adapter_path)


@app.command("create")
def create_sweep(
    config: Path = typer.Option(
        Path("configs/sweep.yaml"),
        "--config",
    ),
    project: str = typer.Option(
        "library-agent",
        "--project",
    ),
) -> None:
    """Register a W&B sweep and print its ID."""
    spec = load_sweep_config(config)

    sweep_id = wandb.sweep(
        sweep=spec.model_dump(mode="python"),
        project=project,
    )

    typer.echo(f"Sweep created: {sweep_id}")


@app.command("agent")
def run_agent(
    sweep_id: str = typer.Option(..., "--sweep-id"),
    base_config: Path = typer.Option(
        Path("configs/training/pilot.yaml"),
        "--base-config",
    ),
    project: str = typer.Option(
        "library-agent",
        "--project",
    ),
    count: int = typer.Option(1, "--count", min=1),
) -> None:
    """Execute a bounded number of W&B sweep trials."""
    # Fail early if the pilot configuration is invalid.
    load_training_config(base_config)

    worker = partial(
        run_trial,
        base_config_path=base_config,
        project=project,
    )

    wandb.agent(
        sweep_id,
        function=worker,
        project=project,
        count=count,
    )


if __name__ == "__main__":
    app()
