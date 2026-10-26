"""Load an MRL encoder checkpoint and run the same evaluation protocol as FlagMRL."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
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


def evaluation_paths(eval_work_dir=None):
    base = Path(eval_work_dir).expanduser().resolve() if eval_work_dir else Path(__file__).resolve().parent
    resolve = lambda path: str((base / path).resolve())
    return {
        "classification": [(resolve(a), resolve(b)) for a, b in TEST_CLS_TASKS],
        "pair": [resolve(path) for path in TEST_PAIR_TASKS],
        "sts": [resolve(path) for path in TEST_STS_TASKS],
    }


def check_eval_data(eval_work_dir=None):
    tasks = evaluation_paths(eval_work_dir)
    specs = [(path, {"text", "label"}) for pair in tasks["classification"] for path in pair]
    specs += [(path, {"sentence1", "sentence2", "label"}) for path in tasks["pair"]]
    specs += [(path, {"sentence1", "sentence2", "score"}) for path in tasks["sts"]]
    errors = []
    for path, columns in specs:
        try:
            frame = pd.read_csv(path, nrows=2)
            missing = columns - set(frame.columns)
            if missing or frame.empty:
                errors.append(f"{path}: missing columns {sorted(missing)} or empty dataset")
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise ValueError("Evaluation CSV checks failed:\n" + "\n".join(errors))
    print(f"Evaluation data ready: {len(specs)} CSV files.")


def evaluate_model(model, tokenizer, dims, output_dir, epoch=None, eval_work_dir=None, batch_size=64):
    """Evaluate the live encoder after each epoch, using the notebook task lists."""
    tasks = evaluation_paths(eval_work_dir)
    options = {"batch_size": batch_size}
    was_training = model.training
    try:
        results = json_safe({
            "checkpoint_dir": str(Path(output_dir).resolve()), "epoch": epoch,
            "matryoshka_dims": list(dims), "pooling": getattr(model, "embedding_pooling", "cls"),
            "classification": eval_classification_task(model, tokenizer, tasks["classification"], dims, **options),
            "pair": eval_pair_task(model, tokenizer, tasks["pair"], dims, **options),
            "sts": eval_sts_task(model, tokenizer, tasks["sts"], dims, **options),
        })
    finally:
        model.train(was_training)
    directory = Path(output_dir)
    names = ["result.json"] + ([f"result_epoch_{epoch}.json"] if epoch is not None else [])
    for name in names:
        destination = directory / name
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
    with (directory / "eval.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(results, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Evaluation results saved to: {directory / 'result.json'}")
    return results


def evaluate(checkpoint_dir, matryoshka_dims=None, device=None, eval_work_dir=None, batch_size=64):
    check_eval_data(eval_work_dir)
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    model = AutoModel.from_pretrained(checkpoint_dir)
    config_path = checkpoint_dir / "training_config.json"
    configuration = json.loads(config_path.read_text()) if config_path.is_file() else {}
    pooling = configuration.get("pooling", "cls")
    if pooling not in {"cls", "mean"}:
        raise ValueError("Unsupported checkpoint pooling.")
    model.embedding_pooling = pooling
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
    return evaluate_model(model, tokenizer, dims, checkpoint_dir,
                          eval_work_dir=eval_work_dir, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval_work_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--check_data", action="store_true")
    args = parser.parse_args()
    if args.check_data:
        check_eval_data(args.eval_work_dir)
    else:
        if not args.checkpoint_dir:
            parser.error("Provide --checkpoint_dir")
        del args.check_data
        evaluate(**vars(args))
