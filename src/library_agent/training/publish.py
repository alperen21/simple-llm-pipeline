"""Publish model artifacts to the Hugging Face Hub."""

from pathlib import Path

from huggingface_hub import HfApi

from library_agent.schemas import DeploymentConfig


def require_file(path: Path) -> Path:
    """Return a validated file path."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def publish_adapter(
    config: DeploymentConfig,
    model_card: Path,
) -> str:
    """Publish the trained LoRA adapter and model card."""
    adapter_dir = config.adapter_dir.resolve()

    require_file(adapter_dir / "adapter_config.json")
    require_file(adapter_dir / "adapter_model.safetensors")
    require_file(model_card)

    api = HfApi()
    repo_id = config.hf_adapter_repo

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not config.hf_public,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        commit_message="Upload evaluated LoRA adapter",
    )

    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
    )

    return f"https://huggingface.co/{repo_id}"


def publish_gguf(
    config: DeploymentConfig,
    model_card: Path,
    license_file: Path,
    notice_file: Path,
) -> str:
    """Publish the selected quantized GGUF and documentation."""
    gguf = config.gguf_dir.resolve() / f"model-{config.quantization}.gguf"

    files = {
        gguf.name: require_file(gguf),
        "README.md": require_file(model_card),
        "LICENSE.txt": require_file(license_file),
        "NOTICE": require_file(notice_file),
    }

    api = HfApi()
    repo_id = config.hf_gguf_repo

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not config.hf_public,
        exist_ok=True,
    )

    for destination, source in files.items():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(source),
            path_in_repo=destination,
        )

    return f"https://huggingface.co/{repo_id}"
