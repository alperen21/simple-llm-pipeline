"""Unit tests for W&B sweep configuration handling."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from library_agent.schemas.training import TrainingConfig
from library_agent.training.sweep import build_trial_config


@pytest.fixture
def base_config(tmp_path: Path) -> TrainingConfig:
    """Create a valid pilot training configuration."""
    return TrainingConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        train_path=tmp_path / "train.jsonl",
        validation_path=tmp_path / "val.jsonl",
        output_dir=tmp_path / "pilot",
    )


def test_sweep_overrides_hyperparameters(
    base_config: TrainingConfig,
) -> None:
    """Sweep parameters override the pilot defaults."""
    trial = build_trial_config(
        base=base_config,
        parameters={
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 0.0001,
        },
        run_id="test123",
        project="library-agent",
    )

    assert trial.lora_r == 32
    assert trial.lora_alpha == 64
    assert trial.learning_rate == 0.0001
    assert trial.wandb_enabled is True


def test_sweep_creates_unique_output_directory(
    base_config: TrainingConfig,
) -> None:
    """Each run gets its own checkpoint directory."""
    trial = build_trial_config(
        base=base_config,
        parameters={"lora_r": 16},
        run_id="unique123",
        project="library-agent",
    )

    assert trial.output_dir == (base_config.output_dir / "unique123")


def test_invalid_rank_rejected(
    base_config: TrainingConfig,
) -> None:
    """Pydantic validation must catch invalid sweep values."""
    with pytest.raises(ValidationError):
        build_trial_config(
            base=base_config,
            parameters={"lora_r": -1},
            run_id="invalid",
            project="library-agent",
        )
