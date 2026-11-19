"""Validated model export and deployment configuration."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuantizationType = Literal["Q4_K_M", "Q5_K_M", "Q8_0"]


class DeploymentConfig(BaseModel):
    """Configuration for model packaging and deployment."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(min_length=1)

    adapter_dir: Path
    merged_dir: Path
    gguf_dir: Path
    llama_cpp_dir: Path

    quantization: QuantizationType = "Q4_K_M"

    ollama_model: str = "library-agent:q4"

    context_length: int = Field(default=4096, ge=256)

    hf_adapter_repo: str = Field(min_length=1)
    hf_gguf_repo: str = Field(min_length=1)
    hf_public: bool = False
