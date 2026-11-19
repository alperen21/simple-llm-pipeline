# 0002 — Cost-aware model selection

- Status: accepted
- Date: 2026-11-19

## Context

Sweeping LoRA hyperparameters on eval_loss is not enough to pick a deployment
model. Latency, memory, tool-call quality, and serving cost differ across
quantizations and hardware.

## Decision

Selection is a pure function over measured `CandidateBenchmark` records:

1. Reject candidates that fail quality or resource gates in `SelectionConfig`.
2. Among the rest, minimize projected USD for `expected_requests`.
3. Persist an auditable `SelectionReport` (eligible scores and reject reasons).

W&B sweeps only optimize `selection/eval_loss`. They do not replace this step.

## Consequences

Someone must produce the benchmark JSON from a real eval suite. The selector
will not train or query models. Hardware and eval-suite identifiers must match
exactly between the JSON and `configs/selection.yaml`.
