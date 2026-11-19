# Training

## Install dependencies

```bash 
uv sync --group training
```

verify the cli is registered

```bash 
uv run --group training library-train --help
```

## Prepare the dataset

```bash
uv run --group training library-train prepare run
```

This downsloads the dataset from huggingface using HfApi. 
Then reads the conversations 
Then Create train/validation/test splits
Each of the data examples are validated using PyDantic
Save reproducibility metadata

## Hyperparameter Tuning with Small LLaMa

Register a LoRA hyperparameter search with Weights & Biases
```bash
uv run --group training library-train sweep create \
  --config configs/sweep.yaml
```

Execute the sweep
```bash
uv run --group training library-train sweep agent \
  --sweep-id YOUR_SWEEP_ID \
  --count 10
```
This basically tries different LoRA configurations and selects the best


Finally, select the best model:
uv run --group training library-train select run \
  --config configs/selection.yaml

Based on the findings update: configs/training/final.yaml

## Train the model

```bash
uv run --group training library-train run \
  --config configs/training/final.yaml
```

This saves the LoRA adapter


## Serve

Deployment is a separate CLI from training. Merge the LoRA adapter with the base model:

```bash
uv run --group training library-deploy merge
```

Convert to GGUF, then quantize:

```bash
uv run --group training library-deploy convert
uv run --group training library-deploy quantize
```

Register with Ollama:

```bash
uv run --group training library-deploy ollama
```

Publish to Hugging Face:

```bash
uv run --group training library-deploy publish-adapter --model-card PATH
uv run --group training library-deploy publish-gguf --model-card PATH --license PATH --notice PATH
```

Or run merge through Ollama registration in one step:

```bash
uv run --group training library-deploy run
```