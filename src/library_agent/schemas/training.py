"""Validated configuration models for supervised fine-tuning."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """Configuration for a reproducible LoRA training run."""

    model_config = ConfigDict(extra="forbid")

    # Model and data
    model_id: str = Field(min_length=1)
    train_path: Path
    validation_path: Path
    output_dir: Path

    # LoRA
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)

    # Optimization
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_length: int = Field(default=1024, ge=128)
    max_steps: int = Field(default=-1, ge=-1)

    # Reproducibility
    seed: int = Field(default=42, ge=0)

    # Logging
    logging_steps: int = Field(default=10, ge=1)
    wandb_enabled: bool = False
    wandb_project: str = "library-agent"
