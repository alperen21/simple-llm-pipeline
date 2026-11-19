"""Unit tests for compute-aware model selection."""

from decimal import Decimal
from pathlib import Path

import pytest

from library_agent.schemas.experiments import (
    CandidateBenchmark,
    SelectionConfig,
)
from library_agent.training.select import (
    projected_cost,
    select_model,
)


@pytest.fixture
def selection_config(tmp_path: Path) -> SelectionConfig:
    """Create deterministic selection criteria."""
    return SelectionConfig(
        benchmark_path=tmp_path / "candidates.json",
        output_path=tmp_path / "selection.json",
        evaluation_suite="test-v1",
        hardware_profile="test-gpu",
        expected_requests=100_000,
        min_task_success_rate=0.85,
        min_tool_call_accuracy=0.90,
        max_p95_latency_ms=5000,
        max_peak_memory_gb=16,
    )


def make_candidate(
    candidate_id: str,
    marginal_cost: str,
    success: float = 0.90,
) -> CandidateBenchmark:
    """Create a synthetic candidate for unit tests."""
    return CandidateBenchmark(
        candidate_id=candidate_id,
        model_id=f"test/{candidate_id}",
        model_revision="test-revision",
        quantization="Q4_K_M",
        evaluation_suite="test-v1",
        hardware_profile="test-gpu",
        experiment_id="test-experiment",
        task_success_rate=success,
        tool_call_accuracy=0.95,
        p95_latency_ms=1000,
        peak_memory_gb=4,
        fine_tuning_cost_usd=Decimal("10"),
        fixed_serving_cost_usd=Decimal("20"),
        marginal_cost_per_request_usd=Decimal(marginal_cost),
    )


def test_projected_cost() -> None:
    """Compute full projected lifecycle cost."""
    model = make_candidate("small", "0.001")

    result = projected_cost(
        model,
        expected_requests=100_000,
    )

    assert result == Decimal("130.000")


def test_select_cheapest_eligible_model(
    selection_config: SelectionConfig,
) -> None:
    """Select the lowest-cost model meeting all constraints."""
    small = make_candidate("small", "0.001")
    large = make_candidate("large", "0.004")

    report = select_model(
        [large, small],
        selection_config,
    )

    assert report.selected.candidate_id == "small"


def test_reject_low_quality_model(
    selection_config: SelectionConfig,
) -> None:
    """Ineligible cheap models must not be selected."""
    small = make_candidate(
        "small",
        "0.0001",
        success=0.70,
    )

    large = make_candidate(
        "large",
        "0.004",
        success=0.95,
    )

    report = select_model(
        [small, large],
        selection_config,
    )

    assert report.selected.candidate_id == "large"
    assert "small" in report.rejected


def test_no_eligible_model(
    selection_config: SelectionConfig,
) -> None:
    """Raise when all candidates fail the constraints."""
    model = make_candidate(
        "low-quality",
        "0.001",
        success=0.50,
    )

    with pytest.raises(
        ValueError,
        match="No eligible models",
    ):
        select_model([model], selection_config)
