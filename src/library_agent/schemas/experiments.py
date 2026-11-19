"""Validated experiment and model-selection schemas."""

from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictModel(BaseModel):
    """Base schema that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------
# Weights & Biases sweep configuration
# --------------------------------------------------


class SweepParameter(StrictModel):
    """Discrete values to explore for one hyperparameter."""

    values: list[int | float] = Field(min_length=1)


class SweepMetric(StrictModel):
    """Optimization objective for a W&B sweep."""

    name: Literal["selection/eval_loss"]
    goal: Literal["minimize"]


class SweepConfig(StrictModel):
    """Validated W&B sweep specification."""

    name: str = "pilot-lora"
    method: Literal["grid", "random", "bayes"] = "random"
    metric: SweepMetric
    parameters: dict[str, SweepParameter]

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        """Reject unsupported hyperparameters."""
        allowed = {
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
        }

        unknown = set(self.parameters) - allowed

        if unknown:
            raise ValueError(f"Unsupported sweep parameters: {sorted(unknown)}")

        if not self.parameters:
            raise ValueError("Sweep parameters cannot be empty.")

        return self


# --------------------------------------------------
# Compute-aware model selection
# --------------------------------------------------


class CandidateBenchmark(StrictModel):
    """Measured performance and cost of one model candidate."""

    candidate_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    quantization: str = Field(min_length=1)

    # Provenance
    evaluation_suite: str = Field(min_length=1)
    hardware_profile: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)

    # Measured model performance
    task_success_rate: float = Field(ge=0, le=1)
    tool_call_accuracy: float = Field(ge=0, le=1)
    p95_latency_ms: float = Field(gt=0)
    peak_memory_gb: float = Field(gt=0)

    # Measured or explicitly estimated costs
    fine_tuning_cost_usd: Decimal = Field(ge=0)
    fixed_serving_cost_usd: Decimal = Field(ge=0)
    marginal_cost_per_request_usd: Decimal = Field(ge=0)


class SelectionConfig(StrictModel):
    """Workload assumptions and hard selection constraints."""

    benchmark_path: Path
    output_path: Path

    evaluation_suite: str
    hardware_profile: str

    expected_requests: int = Field(ge=1)

    min_task_success_rate: float = Field(ge=0, le=1)
    min_tool_call_accuracy: float = Field(ge=0, le=1)

    max_p95_latency_ms: float = Field(gt=0)
    max_peak_memory_gb: float = Field(gt=0)


class CandidateScore(StrictModel):
    """Cost estimate for an eligible model candidate."""

    candidate_id: str
    projected_total_cost_usd: Decimal


class SelectionReport(StrictModel):
    """Auditable result of model selection."""

    selected: CandidateBenchmark
    selected_total_cost_usd: Decimal

    expected_requests: int

    eligible: list[CandidateScore]
    rejected: dict[str, list[str]]
