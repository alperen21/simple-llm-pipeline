# Deployment

Packaging is a **separate** CLI from training: `library-deploy`. Commands still
need the `training` extra because merge and publish import PEFT, Transformers,
and `huggingface_hub`.

```bash
uv run --group training library-deploy --help
```

Default config: `configs/serving/deployment.yaml` (`DeploymentConfig`).

## Prerequisites

- A trained adapter at `adapter_dir` (`adapter_config.json` +
  `adapter_model.safetensors`).
- `base_model` **must** match the adapter's PEFT `base_model_name_or_path`.
- A cloned and built llama.cpp tree at `llama_cpp_dir`, including
  `convert_hf_to_gguf.py` and `build/bin/llama-quantize` (or
  `build/bin/Release/llama-quantize`).
- `conversion_python`: interpreter that can run the converter (defaults to
  `external/llama.cpp/.venv/bin/python`).

Allowed quantizations: `Q4_K_M`, `Q5_K_M`, `Q8_0`.

## Step by step

Merge LoRA into the base Transformers model:

```bash
uv run --group training library-deploy merge
```

Convert the merged directory to FP16 GGUF (`gguf_dir/model-f16.gguf`):

```bash
uv run --group training library-deploy convert
```

Quantize to `gguf_dir/model-{quantization}.gguf`:

```bash
uv run --group training library-deploy quantize
```

Register the quantized file with a local Ollama daemon (`ollama` on `PATH`, API
on `127.0.0.1:11434`):

```bash
uv run --group training library-deploy ollama
```

Publish (requires Hugging Face credentials and model cards):

```bash
uv run --group training library-deploy publish-adapter --model-card PATH
uv run --group training library-deploy publish-gguf \
  --model-card PATH --license PATH --notice PATH
```

Replace `YOUR_USERNAME` in `hf_adapter_repo` and `hf_gguf_repo` before publish.
Repos are created private unless `hf_public: true`.

## One-shot local path

Merge → convert → quantize → Ollama (no Hub upload):

```bash
uv run --group training library-deploy run
```

## Related code

- [`library_agent.serving.cli`](reference.md#library_agent.serving.cli)
- [`library_agent.training.merge`](reference.md#library_agent.training.merge)
- [`library_agent.serving.export`](reference.md#library_agent.serving.export)
- [`library_agent.serving.ollama`](reference.md#library_agent.serving.ollama)
- [`library_agent.training.publish`](reference.md#library_agent.training.publish)
