"""Cost-aware selection of fine-tuned model candidates."""

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import typer
import yaml
from pydantic import TypeAdapter

from library_agent.schemas.experiments import (
    CandidateBenchmark,
    CandidateScore,
    SelectionConfig,
    SelectionReport,
)

app = typer.Typer(help="Select a deployment model.")


def load_selection_config(path: Path) -> SelectionConfig:
    """Load validated selection criteria."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return SelectionConfig.model_validate(data)


def load_benchmarks(path: Path) -> list[CandidateBenchmark]:
    """Load measured candidate benchmarks from JSON."""
    if not path.is_file():
        raise FileNotFoundError(path)

    adapter = TypeAdapter(list[CandidateBenchmark])

    candidates = adapter.validate_json(path.read_text(encoding="utf-8"))

    if not candidates:
        raise ValueError("No candidate benchmarks provided.")

    ids = [candidate.candidate_id for candidate in candidates]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs detected.")

    return candidates


def projected_cost(
    candidate: CandidateBenchmark,
    expected_requests: int,
) -> Decimal:
    """Estimate training plus deployment lifecycle cost."""
    return (
        candidate.fine_tuning_cost_usd
        + candidate.fixed_serving_cost_usd
        + (candidate.marginal_cost_per_request_usd * Decimal(expected_requests))
    )


def check_constraints(
    candidate: CandidateBenchmark,
    config: SelectionConfig,
) -> list[str]:
    """Return all reasons a candidate is ineligible."""
    reasons: list[str] = []

    if candidate.evaluation_suite != config.evaluation_suite:
        reasons.append("Incompatible evaluation suite")

    if candidate.hardware_profile != config.hardware_profile:
        reasons.append("Incompatible benchmark hardware")

    if candidate.task_success_rate < config.min_task_success_rate:
        reasons.append("Task success below threshold")

    if candidate.tool_call_accuracy < config.min_tool_call_accuracy:
        reasons.append("Tool accuracy below threshold")

    if candidate.p95_latency_ms > config.max_p95_latency_ms:
        reasons.append("p95 latency exceeds limit")

    if candidate.peak_memory_gb > config.max_peak_memory_gb:
        reasons.append("Memory requirement exceeds limit")

    return reasons


def select_model(
    candidates: Sequence[CandidateBenchmark],
    config: SelectionConfig,
) -> SelectionReport:
    """Select the lowest-cost model satisfying all constraints.

    Args:
        candidates: Candidate models with measured evaluation results.
        config: Workload assumptions and quality requirements.

    Returns:
        Report containing the selected model and ranked alternatives.

    Raises:
        ValueError: If no candidate satisfies the constraints.
    """
    if not candidates:
        raise ValueError("No candidate benchmarks provided.")

    eligible: list[tuple[CandidateBenchmark, Decimal]] = []
    rejected: dict[str, list[str]] = {}

    seen: set[str] = set()

    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ValueError(f"Duplicate candidate: {candidate.candidate_id}")

        seen.add(candidate.candidate_id)

        reasons = check_constraints(candidate, config)

        if reasons:
            rejected[candidate.candidate_id] = reasons
            continue

        cost = projected_cost(
            candidate,
            config.expected_requests,
        )

        eligible.append((candidate, cost))

    if not eligible:
        details = "; ".join(f"{name}: {', '.join(reasons)}" for name, reasons in rejected.items())

        raise ValueError(f"No eligible models. Rejections: {details}")

    # Minimize cost. Break ties using quality and latency.
    eligible.sort(
        key=lambda item: (
            item[1],
            -item[0].task_success_rate,
            item[0].p95_latency_ms,
            item[0].candidate_id,
        )
    )

    winner, winner_cost = eligible[0]

    return SelectionReport(
        selected=winner,
        selected_total_cost_usd=winner_cost,
        expected_requests=config.expected_requests,
        eligible=[
            CandidateScore(
                candidate_id=candidate.candidate_id,
                projected_total_cost_usd=cost,
            )
            for candidate, cost in eligible
        ],
        rejected=rejected,
    )


def save_report(
    report: SelectionReport,
    path: Path,
) -> None:
    """Serialize the selection result for downstream workflows."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


@app.command("run")
def run_selection(
    config: Path = typer.Option(
        Path("configs/selection.yaml"),
        "--config",
    ),
) -> None:
    """Select a model from measured candidate benchmarks."""
    settings = load_selection_config(config)

    candidates = load_benchmarks(settings.benchmark_path)

    report = select_model(candidates, settings)

    save_report(report, settings.output_path)

    typer.echo(f"Selected model: {report.selected.model_id}")
    typer.echo(f"Quantization: {report.selected.quantization}")
    typer.echo(f"Projected cost: ${report.selected_total_cost_usd:,.2f}")
    typer.echo(f"Report saved: {settings.output_path}")


if __name__ == "__main__":
    app()
