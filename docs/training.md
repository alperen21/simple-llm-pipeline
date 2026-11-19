# Training

Fine-tuning is the `library-train` CLI. Heavy dependencies live in the
`training` extra (`torch`, `transformers`, `trl`, `peft`, `datasets`, `wandb`).

## Install and verify

```bash
uv sync --group training
uv run --group training library-train --help
```

On macOS x86_64, recent `torch` wheels may not resolve. Use a platform that
publishes a matching wheel, or pin a compatible torch build, before running SFT.

## Prepare the dataset

```bash
uv run --group training library-train prepare run
```

This command:

1. Downloads `Archangel-system/glaive-function-calling-v2-openai-native` via
   `huggingface_hub`.
2. Parses conversations and validates each example with
   `library_agent.schemas.dataset.Conversation`.
3. Rejects tool calls to undeclared functions and orphan tool responses.
4. Splits train / validation / test (10% of original train held out for
   validation) with a fixed seed.
5. Writes JSONL under `data/processed/` and a SHA-256 manifest under
   `data/manifests/` for reproducibility.

SFT rows use columns `prompt`, `completion`, and `tools`. Empty datasets and
missing columns fail in `train.load_training_dataset`.

## Hyperparameter sweep

`configs/sweep.yaml` searches discrete LoRA and learning-rate values. The
objective is `selection/eval_loss` (minimize). Allowed parameter names are
`lora_r`, `lora_alpha`, `lora_dropout`, and `learning_rate`.

```bash
uv run --group training library-train sweep create \
  --config configs/sweep.yaml \
  --project library-agent
```

The command prints a W&B sweep ID. Run a bounded number of trials:

```bash
uv run --group training library-train sweep agent \
  --sweep-id YOUR_SWEEP_ID \
  --base-config configs/training/pilot.yaml \
  --count 10
```

Each trial copies the base `TrainingConfig`, overrides the sampled
hyperparameters, writes checkpoints under `output_dir / run_id`, trains through
the same `SFTTrainer` path, reads `metrics.json`, and logs
`selection/eval_loss`.

## Select a candidate

```bash
uv run --group training library-train select run \
  --config configs/selection.yaml
```

Selection loads measured `CandidateBenchmark` records from JSON, drops
candidates that miss quality, latency, or memory gates, then chooses the lowest
projected lifecycle cost for `expected_requests`. The report is written to
`artifacts/selection.json`.

Copy the chosen LoRA settings into `configs/training/final.yaml` before the
full train.

## Train

```bash
uv run --group training library-train run \
  --config configs/training/pilot.yaml
```

`TrainingConfig` fields include model id, dataset paths, LoRA rank/alpha/dropout,
optimizer settings, `max_length`, seed, and optional W&B. Training uses
completion-only loss, gradient checkpointing, and epoch-level eval. The adapter
and tokenizer are saved to `{output_dir}/adapter`.

Pilot config (`configs/training/pilot.yaml`) targets
`meta-llama/Llama-3.2-1B-Instruct` with r=16, α=32, two epochs, batch size 2.

## Related code

- [`library_agent.training.prepare`](reference.md#library_agent.training.prepare)
- [`library_agent.training.sweep`](reference.md#library_agent.training.sweep)
- [`library_agent.training.select`](reference.md#library_agent.training.select)
- [`library_agent.training.train`](reference.md#library_agent.training.train)
- [`library_agent.schemas.training.TrainingConfig`](reference.md#library_agent.schemas.training.TrainingConfig)
