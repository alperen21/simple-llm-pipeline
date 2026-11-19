"""Prepare public tool-calling data for supervised fine-tuning."""

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import typer
from huggingface_hub import HfApi, hf_hub_download
from pydantic import ValidationError

from library_agent.schemas.dataset import Conversation

app = typer.Typer(help="Prepare public training datasets.")

DATASET_ID = "Archangel-system/glaive-function-calling-v2-openai-native"

OUTPUT_DIR = Path("data/processed")
MANIFEST_DIR = Path("data/manifests")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records into memory."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def split_dataset(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
    train_limit: int,
    val_limit: int,
    test_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create reproducible training, validation, and test splits."""
    rng = random.Random(seed)

    # Shuffle entire conversations, not individual messages.
    rng.shuffle(train_records)
    rng.shuffle(test_records)

    # Reserve 10% of the original training data for validation.
    val_size = int(len(train_records) * validation_fraction)

    validation = train_records[:val_size]
    training = train_records[val_size:]

    # Keep small subsets for pilot experiments.
    return {
        "train": training[:train_limit],
        "val": validation[:val_limit],
        "test": test_records[:test_limit],
    }


def validate_tools(conversation: Conversation) -> None:
    """Check that tool calls reference declared tools."""
    declared: set[str] = set()

    for tool in conversation.tools:
        function = tool.get("function")

        if not isinstance(function, dict):
            raise ValueError("Invalid tool definition.")

        name = function.get("name")
        parameters = function.get("parameters")

        if not isinstance(name, str):
            raise ValueError("Invalid function name.")

        if not isinstance(parameters, dict):
            raise ValueError("Invalid tool parameters.")

        declared.add(name)

    issued_calls: set[str] = set()

    for message in conversation.messages:
        for call in message.tool_calls:
            if call.function.name not in declared:
                raise ValueError("Tool call references undeclared function.")

            if call.id:
                issued_calls.add(call.id)

        if (
            message.role == "tool"
            and message.tool_call_id
            and message.tool_call_id not in issued_calls
        ):
            raise ValueError("Orphan tool response.")


def conversation_to_examples(
    conversation: Conversation,
) -> Iterator[dict[str, Any]]:
    """Generate an SFT example for each assistant turn."""
    validate_tools(conversation)

    messages = [
        message.model_dump(
            exclude_none=True,
            exclude_defaults=True,
        )
        for message in conversation.messages
    ]

    for index, message in enumerate(conversation.messages):
        if message.role != "assistant":
            continue

        if not (message.content or message.tool_calls):
            continue

        prompt = messages[:index]

        if not any(m["role"] == "user" for m in prompt):
            continue

        yield {
            "prompt": prompt,
            "completion": [messages[index]],
            "tools": conversation.tools,
        }


def write_examples(
    records: list[dict[str, Any]],
    destination: Path,
) -> dict[str, int]:
    """Validate and export SFT examples as JSONL."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()

    with destination.open("w", encoding="utf-8") as file:
        for record in records:
            try:
                conversation = Conversation.model_validate(record)
                examples = list(conversation_to_examples(conversation))
            except (ValidationError, ValueError, TypeError):
                stats["rejected"] += 1
                continue

            if not examples:
                stats["rejected"] += 1
                continue

            for example in examples:
                file.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["examples"] += 1

            stats["accepted"] += 1

    if stats["examples"] == 0:
        raise ValueError(f"No valid examples produced: {destination}")

    return dict(stats)


@app.command("run")
def prepare(
    seed: int = typer.Option(42, "--seed"),
    train_limit: int = typer.Option(2000, min=1),
    val_limit: int = typer.Option(200, min=1),
    test_limit: int = typer.Option(200, min=1),
    validation_fraction: float = typer.Option(0.1, min=0.01, max=0.5),
) -> None:
    """Download and prepare public tool-calling data."""
    # 1. Pin the dataset revision.
    api = HfApi()
    revision = api.dataset_info(DATASET_ID).sha

    # 2. Download the original train/test JSONL files.
    train_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename="train.jsonl",
            repo_type="dataset",
            revision=revision,
        )
    )

    test_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename="test.jsonl",
            repo_type="dataset",
            revision=revision,
        )
    )

    # 3. Read the conversations.
    train_records = read_jsonl(train_path)
    test_records = read_jsonl(test_path)

    # 4. Create train/validation/test splits.
    splits = split_dataset(
        train_records,
        test_records,
        seed=seed,
        validation_fraction=validation_fraction,
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
    )

    # 5. Convert conversations into SFT examples.
    summary: dict[str, Any] = {}

    for name, records in splits.items():
        output = OUTPUT_DIR / f"{name}.jsonl"

        stats = write_examples(records, output)

        summary[name] = {
            **stats,
            "sha256": sha256_file(output),
        }

        typer.echo(f"{name}: {stats['examples']} examples ({stats.get('rejected', 0)} rejected)")

    # 6. Save reproducibility metadata.
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": DATASET_ID,
        "revision": revision,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_train_sha256": sha256_file(train_path),
        "source_test_sha256": sha256_file(test_path),
        "limits": {
            "train": train_limit,
            "val": val_limit,
            "test": test_limit,
        },
        "splits": summary,
    }

    (MANIFEST_DIR / "dataset.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    typer.echo("Dataset preparation complete.")


if __name__ == "__main__":
    app()
