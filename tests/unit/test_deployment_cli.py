"""Unit tests for deployment CLI configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from library_agent.serving.cli import load_config


def write_config(path: Path, extra: str = "") -> Path:
    """Write a minimal valid deployment YAML file."""
    path.write_text(
        (
            "base_model: meta-llama/Llama-3.2-1B-Instruct\n"
            "adapter_dir: artifacts/pilot/adapter\n"
            "merged_dir: artifacts/merged\n"
            "gguf_dir: artifacts/gguf\n"
            "llama_cpp_dir: external/llama.cpp\n"
            "hf_adapter_repo: user/adapter\n"
            "hf_gguf_repo: user/gguf\n"
            f"{extra}"
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    """A valid mapping becomes a DeploymentConfig."""
    path = write_config(tmp_path / "deploy.yaml")

    config = load_config(path)

    assert config.quantization == "Q4_K_M"
    assert config.ollama_model == "library-agent:q4"


def test_load_config_missing_file(tmp_path: Path) -> None:
    """A missing file is rejected before validation."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    """YAML lists are not valid deployment configs."""
    path = tmp_path / "deploy.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_load_config_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    """Extra keys are forbidden by the schema."""
    path = write_config(
        tmp_path / "deploy.yaml",
        extra="unknown_field: true",
    )

    with pytest.raises(ValidationError):
        load_config(path)
