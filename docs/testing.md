# Testing

```bash
uv run pytest
```

Markers: `integration`, `e2e`, `gpu`. Coverage is configured on `library_agent`
with `fail_under = 90` when you run the coverage report.

## Unit tests

| Test | Covers |
| --- | --- |
| `tests/unit/test_deployment_cli.py` | `serving.cli.load_config` YAML mapping, missing file, extra keys |
| `tests/unit/test_sweep.py` | `build_trial_config` overlays and unknown parameter rejection |
| `tests/unit/test_model_selection.py` | constraint checks and lowest-cost winner |

`tests/conftest.py` is the shared fixture module.

## Manual checks

RAG through MCP (server must already be running, index must exist):

```bash
uv run python -m library_agent.rag.ingest
uv run python -m library_agent.mcp.server   # separate terminal
uv run python test_rag.py
```

`test_rag.py` asserts `search_knowledge` is registered and retrieves passages
for a borrowing-policy query.

GPU SFT and llama.cpp conversion are not exercised in default CI. Mark those
paths `gpu` when you add them.
