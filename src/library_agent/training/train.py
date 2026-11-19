"""LoRA supervised fine-tuning using Hugging Face TRL."""

import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from library_agent.schemas.training import TrainingConfig


def load_training_dataset(path: Path) -> Dataset:
    """Load a prepared JSONL dataset.

    Args:
        path: Path to a normalized SFT dataset.

    Returns:
        Hugging Face Dataset containing the training examples.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or has invalid columns.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = Dataset.from_json(str(path))

    required_columns = {"prompt", "completion", "tools"}

    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"Expected columns {required_columns}, found {dataset.column_names}")

    if len(dataset) == 0:
        raise ValueError("Training dataset cannot be empty.")

    return dataset


def create_lora_config(config: TrainingConfig) -> LoraConfig:
    """Create PEFT's LoRA configuration."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )


def create_sft_config(config: TrainingConfig) -> SFTConfig:
    """Create the supervised training configuration."""
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    fp16 = torch.cuda.is_available() and not bf16

    return SFTConfig(
        output_dir=str(config.output_dir),
        # Optimization
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=(config.gradient_accumulation_steps),
        max_steps=config.max_steps,
        max_length=config.max_length,
        # Dataset and loss
        completion_only_loss=True,
        # Precision
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        # Evaluation and checkpoints
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        # Logging
        logging_steps=config.logging_steps,
        report_to=(["wandb"] if config.wandb_enabled else "none"),
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
    )


def train(config: TrainingConfig) -> Path:
    """Fine-tune a causal language model using LoRA.

    Args:
        config: Validated experiment configuration.

    Returns:
        Path containing the trained LoRA adapter.
    """
    if config.wandb_enabled:
        os.environ.setdefault(
            "WANDB_PROJECT",
            config.wandb_project,
        )

    # 1. Load data
    train_dataset = load_training_dataset(config.train_path)
    validation_dataset = load_training_dataset(config.validation_path)

    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Configure LoRA
    lora_config = create_lora_config(config)

    # 4. Configure training
    sft_config = create_sft_config(config)

    # 5. Initialize trainer
    trainer = SFTTrainer(
        model=config.model_id,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # 6. Train and evaluate
    trainer.train()
    trainer.evaluate()

    # 7. Save the trained adapter
    adapter_path = config.output_dir / "adapter"

    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    return adapter_path
