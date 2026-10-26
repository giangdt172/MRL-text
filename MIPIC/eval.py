"""Evaluate a saved MIPIC encoder and write result.json, as in MRL/FlagMRL."""

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

from MIPIC.config import MODEL_CONFIGS
from MIPIC.data_preparation import TEST_CLS_TASKS, TEST_STS_TASKS, TEST_PAIR_TASKS
from MIPIC.eval_modules import eval_classification_task, eval_pair_task, eval_sts_task


def json_safe(value):
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
    base = Path(eval_work_dir).expanduser().resolve() if eval_work_dir else Path(__file__).resolve().parent.parent
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
    options["pooling"] = getattr(model, "embedding_pooling", "cls")
    was_training = model.training
    try:
        results = json_safe({
            "checkpoint_dir": str(Path(output_dir).resolve()) if output_dir is not None else None, "epoch": epoch,
            "matryoshka_dims": list(dims), "pooling": getattr(model, "embedding_pooling", "cls"),
            "classification": eval_classification_task(model, tokenizer, tasks["classification"], dims, **options),
            "pair": eval_pair_task(model, tokenizer, tasks["pair"], dims, **options),
            "sts": eval_sts_task(model, tokenizer, tasks["sts"], dims, **options),
        })
    finally:
        model.train(was_training)
    if output_dir is None:
        return results
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


def evaluate(checkpoint_dir=None, matryoshka_dims=None, device=None, model_name=None,
             model_key=None, pooling=None, batch_size=64, eval_work_dir=None):
    check_eval_data(eval_work_dir)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    checkpoint = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else None
    source = str(checkpoint) if checkpoint else model_name or MODEL_CONFIGS[model_key]["model_name"]
    configuration_path = checkpoint / "training_config.json" if checkpoint else None
    configuration = json.loads(configuration_path.read_text()) if configuration_path and configuration_path.is_file() else {}
    pooling = pooling or configuration.get("pooling", "cls")
    if pooling not in {"cls", "mean"}:
        raise ValueError("Checkpoint pooling must be cls or mean.")
    model = AutoModel.from_pretrained(source)
    dims = matryoshka_dims or configuration.get("matryoshka_dims")
    if dims is None:
        dims = [d for d in (16, 32, 64, 128, 256, 512) if d < model.config.hidden_size] + [model.config.hidden_size]
    if not dims or any(isinstance(d, bool) or not isinstance(d, int) or not 0 < d <= model.config.hidden_size for d in dims):
        raise ValueError("Evaluation dimensions must be in [1, hidden_size].")
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad_token or eos_token for batched evaluation.")
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select --device cpu for CPU evaluation.")
    model.to(device).eval()
    print(f"Model: {source}\nDims: {dims}\nDevice: {device}\nPooling: {pooling}")
    model.embedding_pooling = pooling
    return evaluate_model(model, tokenizer, dims, checkpoint,
                          eval_work_dir=eval_work_dir, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--model_key", choices=list(MODEL_CONFIGS), default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pooling", choices=["cls", "mean"], default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--check_data", action="store_true", help="Check evaluation CSVs and exit")
    parser.add_argument("--eval_work_dir", default=None)
    args = parser.parse_args()
    if args.check_data:
        check_eval_data(args.eval_work_dir)
    else:
        if not (args.checkpoint_dir or args.model_name or args.model_key):
            parser.error("Provide --checkpoint_dir, --model_name or --model_key.")
        del args.check_data
        evaluate(**vars(args))
