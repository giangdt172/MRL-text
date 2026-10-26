"""Evaluate an original encoder with the same mean-pooling protocol as FlagMRL.

From the repository root:
    python eval_original.py --model_key bert
    python eval_original.py --model_key bgem3 --device cuda
    python eval_original.py --model_name /path/to/local/model --device cpu

Datasets are resolved from --data_root (the repository root by default), so the
script can be run from any working directory. Results go to --output_dir.
"""

import argparse
import json
import math
from pathlib import Path
import sys

# The entry point lives at the repository root; the existing evaluation modules
# still import data_preparation from their own FlagMRL directory.
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "FlagMRL"))

import torch
from transformers import AutoModel, AutoTokenizer

from FlagMRL.config import MODEL_CONFIGS
from data_preparation import TEST_CLS_TASKS, TEST_PAIR_TASKS, TEST_STS_TASKS
from eval_modules import eval_classification_task, eval_pair_task, eval_sts_task


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


def evaluate(model_key=None, model_name=None, matryoshka_dims=None, device=None,
             output_dir=None, data_root=None):
    if (model_key is None) == (model_name is None):
        raise ValueError("Provide exactly one of model_key or model_name.")
    model_name = model_name if model_name is not None else MODEL_CONFIGS[model_key]["model_name"]
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        model_name = str(local_path.resolve())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")

    # Task lists use paths relative to FlagMRL/, as in the trained evaluator.
    data_root = Path(data_root).expanduser().resolve() if data_root is not None else PROJECT_DIR
    def task_path(path):
        return str((data_root / "FlagMRL" / path).resolve())

    cls_tasks = [(task_path(train), task_path(test)) for train, test in TEST_CLS_TASKS]
    pair_tasks = [task_path(path) for path in TEST_PAIR_TASKS]
    sts_tasks = [task_path(path) for path in TEST_STS_TASKS]
    all_paths = [path for pair in cls_tasks for path in pair] + pair_tasks + sts_tasks
    missing = sorted({path for path in all_paths if not Path(path).is_file()})
    if missing:
        raise FileNotFoundError("Evaluation CSV files not found; check --data_root:\n" + "\n".join(missing))

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    full_dim = model.config.hidden_size
    dims = ([d for d in (16, 32, 64, 128, 256, 512) if d < full_dim] + [full_dim]
            if matryoshka_dims is None else list(matryoshka_dims))
    if not dims or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= full_dim for d in dims):
        raise ValueError(f"Evaluation dimensions must be integers in [1, {full_dim}].")
    model.to(device).eval()
    label = model_key or Path(model_name.rstrip("/")).name
    output_dir = (Path(output_dir).expanduser().resolve() if output_dir is not None
                  else PROJECT_DIR / "results" / "original" / label)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Original model: {model_name}\nPooling: mean\nDims: {dims}\nDevice: {device}\nOutput: {output_dir}")
    results = json_safe({
        "model_name": model_name,
        "model_key": model_key,
        "model_type": "original",
        "pooling": "mean",
        "matryoshka_dims": dims,
        "device": str(device),
        "data_root": str(data_root),
        "classification": eval_classification_task(model, tokenizer, cls_tasks, dims),
        "pair": eval_pair_task(model, tokenizer, pair_tasks, dims),
        "sts": eval_sts_task(model, tokenizer, sts_tasks, dims),
    })
    result_path = output_dir / "result.json"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    temporary_path.replace(result_path)
    print(f"Results saved to: {result_path}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_key", choices=list(MODEL_CONFIGS))
    source.add_argument("--model_name", help="Original Hugging Face model name or local encoder directory")
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None,
                        help="Default: 16, 32, 64, 128, 256, 512 below hidden_size, plus hidden_size")
    parser.add_argument("--device", default=None, help="cuda, cuda:N, or cpu; auto-selects when omitted")
    parser.add_argument("--output_dir", default=None, help="Default: results/original/<model_key or model name>")
    parser.add_argument("--data_root", default=None,
                        help="Directory containing multitask-data/ and sts-data/; default: repository root")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(**vars(parse_args()))
