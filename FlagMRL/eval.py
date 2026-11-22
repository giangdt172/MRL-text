"""Load the FlagMRL adapter, then run the existing evaluation protocol unchanged."""

import argparse
import json
import math
from pathlib import Path
import sys

# The copied evaluation modules use sibling imports, as in the MRL baseline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from data_preparation import TEST_CLS_TASKS, TEST_STS_TASKS, TEST_PAIR_TASKS
from eval_modules import eval_classification_task, eval_pair_task, eval_sts_task
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


def evaluate(checkpoint_dir, matryoshka_dims=None, device=None):
    model = FlagTextModel.from_pretrained(checkpoint_dir)
    dims = model.projector.dims if matryoshka_dims is None else matryoshka_dims
    if not dims or any(d < 1 or d > model.config.hidden_size for d in dims):
        raise ValueError("Evaluation dimensions must be in [1, hidden_size].")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    print(f"Checkpoint: {checkpoint_dir}\nDims: {dims}\nDevice: {device}")
    results = {
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "matryoshka_dims": list(dims),
        "classification": eval_classification_task(model, tokenizer, TEST_CLS_TASKS, dims),
        "pair": eval_pair_task(model, tokenizer, TEST_PAIR_TASKS, dims),
        "sts": eval_sts_task(model, tokenizer, TEST_STS_TASKS, dims),
    }
    results = json_safe(results)
    result_path = Path(checkpoint_dir) / "result.json"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    temporary_path.replace(result_path)
    summary = format_summary(results)
    summary_path = Path(checkpoint_dir) / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Results saved to: {result_path.resolve()}")
    print(f"Summary saved to: {summary_path.resolve()}")
    print(f"\nEvaluation summary\n\n{summary}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
