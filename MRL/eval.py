"""Load an MRL encoder checkpoint and run the same evaluation protocol as FlagMRL."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModel, AutoTokenizer

from MRL.data_preparation import TEST_CLS_TASKS, TEST_STS_TASKS, TEST_PAIR_TASKS
from MRL.eval_modules import eval_classification_task, eval_pair_task, eval_sts_task
from MRL.loss import resolve_dims


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


def evaluate(checkpoint_dir, matryoshka_dims=None, device=None):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    model = AutoModel.from_pretrained(checkpoint_dir)
    config_path = checkpoint_dir / "training_config.json"
    configuration = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if configuration.get("pooling", "mean") != "mean":
        raise ValueError("Unsupported checkpoint pooling; evaluation uses mean pooling.")
    dims = matryoshka_dims
    if dims is None:
        dims = configuration.get("matryoshka_dims") or resolve_dims(model.config.hidden_size)
    if not dims or any(isinstance(d, bool) or not isinstance(d, int)
                       or d < 1 or d > model.config.hidden_size for d in dims):
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
    print(f"Results saved to: {result_path.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
