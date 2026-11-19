# Evaluation

Two evaluation surfaces exist today. The rest of `library_agent.evaluation` is
a stub for future gates and agent benchmarks.

## Training metrics

`SFTTrainer.evaluate()` runs after each train. Sweep trials require
`{output_dir}/metrics.json` with `eval_loss`. That value is logged to W&B as
`selection/eval_loss` and is the only allowed sweep metric name.

## Offline model selection

`library-train select` does **not** run the model. It scores
`CandidateBenchmark` JSON produced by an external evaluation suite:

- Same `evaluation_suite` and `hardware_profile` as `configs/selection.yaml`
- Quality floors: `min_task_success_rate`, `min_tool_call_accuracy`
- Ops ceilings: `max_p95_latency_ms`, `max_peak_memory_gb`
- Cost: fine-tune + fixed serving + `expected_requests * marginal_cost`

Rejected candidates appear in the report with reason strings. Eligible
candidates are ranked by projected total USD.

## Planned

`evaluation/metrics.py`, `gates.py`, and `benchmark.py` are reserved for
in-process agent benchmarks (tool-call accuracy, task success, latency). Until
those modules have implementations, do not import them from production paths.
