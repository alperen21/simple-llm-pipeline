"""Merge a trained LoRA adapter with its base model."""

from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from library_agent.schemas import DeploymentConfig


def merge_adapter(config: DeploymentConfig) -> Path:
    """Merge LoRA weights into the original model.

    Args:
        config: Validated deployment configuration.

    Returns:
        Directory containing merged model weights and tokenizer.

    Raises:
        FileNotFoundError: If the adapter is missing.
        ValueError: If the adapter belongs to another base model.
    """
    adapter_dir = config.adapter_dir.resolve()
    output_dir = config.merged_dir.resolve()

    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No PEFT adapter found in {adapter_dir}")

    # Read the model used when the adapter was trained.
    peft_config = PeftConfig.from_pretrained(adapter_dir)

    if peft_config.base_model_name_or_path != config.base_model:
        raise ValueError(
            "Adapter/base-model mismatch: "
            f"{peft_config.base_model_name_or_path} "
            f"!= {config.base_model}"
        )

    # Load the original, unquantized base model.
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype="auto",
        low_cpu_mem_usage=True,
    )

    # Apply the trained adapter.
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
    )

    peft_model.eval()

    # Produce a standalone causal language model.
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Prefer the tokenizer saved alongside the adapter.
    tokenizer_source = (
        str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else config.base_model
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(output_dir)

    return output_dir
