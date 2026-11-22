"""Evaluate a live FlagMRL model after each epoch or load a saved checkpoint."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from FlagMRL.data_preparation import TEST_CLS_TASKS, TEST_STS_TASKS, TEST_PAIR_TASKS
from FlagMRL.eval_modules import eval_classification_task, eval_pair_task, eval_sts_task
from FlagMRL.model import FlagTextModel


def json_safe(value):
    """Convert NumPy scalars and undefined metrics to standard JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def format_summary(results):
    """One two-column table per test; STS reports correlation, not accuracy."""
    tables = []
    for task_type in ("classification", "pair", "sts"):
        metric_name = "Spearman" if task_type == "sts" else "Accuracy"
        for test_name, levels in results.get(task_type, {}).items():
            lines = [f"{test_name} ({task_type})", "",
                     f"| Dims | {metric_name} |", "| ---: | ---: |"]
            for key in sorted(levels, key=lambda key: int(key.removeprefix("dim_"))):
                dim = int(key.removeprefix("dim_"))
                value = levels[key] if task_type == "sts" else levels[key]["accuracy"]
                score = "N/A" if value is None or not math.isfinite(value) else f"{value:.4f}"
                lines.append(f"| {dim} | {score} |")
            tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n"


def evaluation_paths(eval_work_dir=None):
    base = Path(eval_work_dir).expanduser().resolve() if eval_work_dir else Path(__file__).resolve().parent
    return {
        "classification": [(str((base / a).resolve()), str((base / b).resolve()))
                           for a, b in TEST_CLS_TASKS],
        "pair": [str((base / path).resolve()) for path in TEST_PAIR_TASKS],
        "sts": [str((base / path).resolve()) for path in TEST_STS_TASKS],
    }


def evaluate_model(model, tokenizer, dims, output_dir, epoch=None, eval_work_dir=None, batch_size=64):
    """Reuse the live encoder/basis and restore training mode before returning."""
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be positive.")
    paths = evaluation_paths(eval_work_dir)
    was_training = model.training
    try:
        model.eval()
        results = json_safe({
            "checkpoint_dir": str(Path(output_dir).resolve()),
            "epoch": epoch,
            "matryoshka_dims": list(dims),
            "classification": eval_classification_task(model, tokenizer, paths["classification"], dims,
                                                       batch_size=batch_size),
            "pair": eval_pair_task(model, tokenizer, paths["pair"], dims, batch_size=batch_size),
            "sts": eval_sts_task(model, tokenizer, paths["sts"], dims, batch_size=batch_size),
        })
    finally:
        model.train(was_training)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result_text = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    summary = format_summary(results)
    suffixes = [""] + ([f"_epoch_{epoch}"] if epoch is not None else [])
    for suffix in suffixes:
        result_path = directory / f"result{suffix}.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(result_text, encoding="utf-8")
        temporary_path.replace(result_path)
        (directory / f"summary{suffix}.txt").write_text(summary, encoding="utf-8")
    title = "Evaluation summary" if epoch is None else f"Epoch {epoch} evaluation summary"
    with (directory / "eval.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{title}\n\n{summary}\n")
    print(f"Results and summaries saved to: {directory.resolve()}")
    print(f"\n{title}\n\n{summary}")
    return results


def evaluate(checkpoint_dir, matryoshka_dims=None, device=None, eval_work_dir=None, batch_size=64):
    model = FlagTextModel.from_pretrained(checkpoint_dir)
    dims = model.projector.dims if matryoshka_dims is None else matryoshka_dims
    if not dims or any(d < 1 or d > model.config.hidden_size for d in dims):
        raise ValueError("Evaluation dimensions must be in [1, hidden_size].")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    print(f"Checkpoint: {checkpoint_dir}\nDims: {dims}\nDevice: {device}")
    return evaluate_model(model, tokenizer, dims, checkpoint_dir,
                          eval_work_dir=eval_work_dir, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval_work_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    evaluate(**vars(args))
