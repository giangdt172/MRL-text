"""Selected MTEB text tasks evaluated independently at each Matryoshka prefix.

Uses MTEB 2's DataLoader encoder protocol. Pooling matches training; FlagTextModel
applies its learned basis before pooling/truncation. Normalize after truncation.
"""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from common.mteb_tasks import DEFAULT_EVAL_TASKS
from common.mteb_results import format_score, write_result_summaries


def add_eval_args(parser, *, training=False):
    if training:
        parser.add_argument("--run_eval", type=int, choices=[0, 1], default=1,
                            help="Run selected MTEB tasks after each epoch; 0 disables evaluation")
    parser.add_argument("--eval_tasks", nargs="+", default=list(DEFAULT_EVAL_TASKS),
                        help="Override the selected 21-task preset from common/mteb_tasks.py")
    parser.add_argument("--eval_splits", nargs="+", default=["test"])
    parser.add_argument("--eval_languages", nargs="+", default=["eng"])
    parser.add_argument(*(["--eval_batch_size"] if training else ["--eval_batch_size", "--batch_size"]),
                        dest="eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_max_length", type=int, default=256)


def validate_eval_args(args):
    if args.eval_batch_size < 1 or args.eval_max_length < 1:
        raise ValueError("eval_batch_size and eval_max_length must be positive.")
    if getattr(args, "run_eval", True) and not args.eval_tasks:
        raise ValueError("Specify --eval_tasks to select MTEB benchmarks, or use --run_eval 0 for training only.")


def evaluation_dims(full_dim, dims=None):
    dims = list(dims) if dims is not None else [d for d in (16, 32, 64, 128, 256, 512) if d < full_dim] + [full_dim]
    if not dims or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= full_dim for d in dims):
        raise ValueError(f"Evaluation dimensions must be integers in [1, {full_dim}].")
    return sorted(set(dims))


class MatryoshkaEncoder:
    def __init__(self, model, tokenizer, dim, *, pooling="cls", max_length=256, batch_size=64):
        self.model, self.tokenizer, self.dim = model, tokenizer, dim
        self.pooling, self.max_length, self.batch_size = pooling, max_length, batch_size
        evaluation_dims(model.config.hidden_size, [dim])
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean.")
        self.mteb_model_meta = None  # Assigned by evaluate_model; keeps the adapter importable without MTEB.

    @torch.inference_mode()
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        chunks = []
        device = next(self.model.parameters()).device
        # Keep the unprompted training protocol for both queries and passages.
        for batch in inputs:
            texts = batch["text"]
            for start in range(0, len(texts), self.batch_size):
                features = self.tokenizer(list(texts[start:start + self.batch_size]), padding=True,
                                          truncation=True, max_length=self.max_length, return_tensors="pt")
                features = {key: value.to(device) for key, value in features.items()}
                hidden = self.model(**features, return_dict=True).last_hidden_state.float()
                if self.pooling == "cls":
                    embeddings = hidden[:, 0]
                else:
                    mask = features["attention_mask"].unsqueeze(-1).float()
                    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                chunks.append(F.normalize(embeddings[:, :self.dim], dim=-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, self.dim), dtype=np.float32)

    @staticmethod
    def similarity(embeddings1, embeddings2):
        first = F.normalize(torch.as_tensor(embeddings1).float(), dim=-1)
        second = F.normalize(torch.as_tensor(embeddings2).float(), dim=-1)
        return first @ second.T

    @staticmethod
    def similarity_pairwise(embeddings1, embeddings2):
        first = F.normalize(torch.as_tensor(embeddings1).float(), dim=-1)
        second = F.normalize(torch.as_tensor(embeddings2).float(), dim=-1)
        return (first * second).sum(-1)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_model(model, tokenizer, dims, output_dir, *, eval_tasks=None, epoch=None,
                   eval_splits=None, eval_languages=None, batch_size=64, max_length=256,
                   pooling=None, seed=42):
    if eval_tasks is None:
        eval_tasks = DEFAULT_EVAL_TASKS
    if not eval_tasks:
        raise ValueError("Provide explicit MTEB task names in --eval_tasks.")
    if batch_size < 1 or max_length < 1:
        raise ValueError("Evaluation batch_size and max_length must be positive.")
    dims = evaluation_dims(model.config.hidden_size, dims)
    pooling = pooling or getattr(model, "embedding_pooling", "mean" if hasattr(model, "projector") else "cls")
    names = list(dict.fromkeys(eval_tasks))
    splits, languages = list(eval_splits or ["test"]), list(eval_languages or ["eng"])
    directory = Path(output_dir).expanduser().resolve()
    raw_dir = directory / ("mteb" if epoch is None else f"mteb_epoch_{epoch}")
    results = {"backend": "mteb", "mteb_version": version("mteb"), "epoch": epoch,
               "model_name": getattr(model.config, "_name_or_path", ""), "output_dir": str(directory),
               "tasks": [], "eval_splits": splits,
               "eval_languages": languages, "matryoshka_dims": dims, "pooling": pooling,
               "max_length": max_length, "seed": seed, "metrics": {}, "scores": {}, "raw_results": {}}
    was_training = model.training
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        model.eval()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            import mteb
            from mteb.models.model_meta import ModelMeta

            for dim in dims:
                # Each level starts with the same evaluation seed, independent of task side effects.
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                # Fresh task instances also reset MTEB's task-local sampling RNGs.
                tasks = [mteb.get_task(name, eval_splits=splits, languages=languages) for name in names]
                results["tasks"] = [task.metadata.name for task in tasks]
                results["metrics"] = {task.metadata.name: task.metadata.main_score for task in tasks}
                adapter = MatryoshkaEncoder(model, tokenizer, dim, pooling=pooling,
                                            max_length=max_length, batch_size=batch_size)
                adapter.mteb_model_meta = ModelMeta.create_empty(overwrites={
                    "name": f"local/{directory.name}", "revision": f"epoch-{epoch}-dim-{dim}",
                    "embed_dim": dim, "max_tokens": max_length,
                    "modalities": ["text"], "similarity_fn_name": "cosine"})
                # Never reuse scores from a different epoch, checkpoint, or dimension.
                # MTEB forwards num_proc to datasets.load_dataset; 0 causes division by zero there.
                model_result = mteb.evaluate(adapter, tasks=tasks, cache=None, overwrite_strategy="always",
                                             co2_tracker=False, raise_error=True, num_proc=None,
                                             encode_kwargs={"batch_size": batch_size})
                key = f"dim_{dim}"
                results["scores"][key] = {task.task_name: task.get_score() for task in model_result.task_results}
                raw_path = raw_dir / key / "results.json"
                write_json(raw_path, model_result.model_dump(mode="json"))
                results["raw_results"][key] = str(raw_path.relative_to(directory))
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    suffixes = [""] + ([f"_epoch_{epoch}"] if epoch is not None else [])
    rows = ["| Task | Metric | " + " | ".join(str(dim) for dim in dims) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in dims) + " |"]
    for name in results["tasks"]:
        rows.append("| " + name + " | " + results["metrics"][name] + " | " + " | ".join(
            format_score(results["scores"][f"dim_{dim}"][name]) for dim in dims) + " |")
    label = f"epoch {epoch}" if epoch is not None else "standalone evaluation"
    summary = f"MTEB main_score by embedding dimension ({label})\n\n" + "\n".join(rows) + "\n"
    for suffix in suffixes:
        write_json(directory / f"result{suffix}.json", results)
        (directory / f"summary{suffix}.txt").write_text(summary)
    write_result_summaries(directory)
    print(summary)
    print(f"MTEB results: {directory / 'result.json'}")
    print(f"All epochs: {directory / 'summary_all_epochs.txt'} | CSV: {directory / 'results.csv'}")
    return json_safe(results)


def evaluate(checkpoint_dir=None, *, method="mrl", model_name=None, model_key=None,
             matryoshka_dims=None, device=None, output_dir=None, pooling=None,
             eval_tasks=None, eval_splits=None, eval_languages=None,
             eval_batch_size=64, eval_max_length=256, seed=42):
    if eval_tasks is None:
        eval_tasks = DEFAULT_EVAL_TASKS
    if not eval_tasks:
        raise ValueError("Provide --eval_tasks with the selected MTEB task names.")
    checkpoint = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else None
    config_path = checkpoint / "training_config.json" if checkpoint else None
    configuration = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
    if checkpoint and (checkpoint / "flag_config.json").is_file():
        from FlagMRL.model import FlagTextModel

        model = FlagTextModel.from_pretrained(checkpoint)
        source = str(checkpoint)
        pooling = "mean"
        if matryoshka_dims is None:
            matryoshka_dims = model.projector.dims
    else:
        if not checkpoint and not model_name:
            if not model_key:
                raise ValueError("Provide a checkpoint, model_name, or model_key.")
            from MRL.config import MODEL_CONFIGS

            model_name = MODEL_CONFIGS[model_key]["model_name"]
        source = str(checkpoint) if checkpoint else model_name
        model = AutoModel.from_pretrained(source)
        pooling = pooling or configuration.get("pooling", "mean" if method in {"ese", "flag_mrl", "original"} else "cls")
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token for batching.")
        tokenizer.pad_token = tokenizer.eos_token
    dims = evaluation_dims(model.config.hidden_size, matryoshka_dims if matryoshka_dims is not None
                           else configuration.get("matryoshka_dims"))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    model.embedding_pooling = pooling
    output_dir = output_dir or checkpoint or Path("results") / "original" / (model_key or Path(source).name)
    return evaluate_model(model, tokenizer, dims, output_dir, eval_tasks=eval_tasks,
                          eval_splits=eval_splits, eval_languages=eval_languages,
                          batch_size=eval_batch_size, max_length=eval_max_length, pooling=pooling, seed=seed)


def main(method="mrl", argv=None):
    from MRL.config import MODEL_CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint_dir")
    source.add_argument("--model_name")
    source.add_argument("--model_key", choices=list(MODEL_CONFIGS))
    parser.add_argument("--matryoshka_dims", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pooling", choices=["cls", "mean"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_eval_args(parser)
    args = parser.parse_args(argv)
    validate_eval_args(args)
    return evaluate(method=method, **vars(args))
