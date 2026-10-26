import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, get_scheduler

from MRL.config import MODEL_CONFIGS
from MRL.data_preparation import build_train_loader
from MRL.loss import Matry_infonce, info_nce, resolve_dims


def batch_inputs(batch, side, device):
    return {key: batch[f"{key}{side}_stu"].to(device, non_blocking=device.type == "cuda")
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if f"{key}{side}_stu" in batch}

def mean_pooling(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("epochs, batch_size and max_length must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be positive and finite.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite.")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be nonnegative and finite (0 disables clipping).")
    if not math.isfinite(args.min_lr) or not 0 <= args.min_lr <= args.lr:
        raise ValueError("min_lr must be finite and between zero and lr.")
    if args.num_workers < 0 or args.eval_batch_size < 1:
        raise ValueError("num_workers must be nonnegative; eval_batch_size must be positive.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    if args.log_every < 1:
        raise ValueError("log_every must be positive.")
    if args.no_amp and args.precision not in {"auto", "fp32"}:
        raise ValueError("--no_amp cannot be combined with --precision fp16/bf16.")
    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {data_path}. Set --data_path to your CSV.")
    if "text" not in pd.read_csv(data_path, nrows=0).columns:
        raise ValueError(f"Training CSV must contain a 'text' column: {data_path}")
    args.data_path = str(data_path)
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())


def resolve_device_precision(device_name=None, precision="auto", no_amp=False):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported training devices: cpu, cuda, or cuda:N.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Check your GPU/PyTorch CUDA installation.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"Invalid CUDA index {index}; visible GPUs: {torch.cuda.device_count()}.")
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    if no_amp:
        precision = "fp32"
    if precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16.")
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU training uses fp32. Select --precision fp32 or auto.")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16. Use --precision fp16 or auto.")
    return device, precision


def configure_logging(output_dir):
    logger = logging.getLogger("MRL.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log", mode="w")):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def train(args):
    validate_args(args)
    if args.run_eval:
        from MRL.eval import check_eval_data
        check_eval_data(args.eval_work_dir)
    device, precision = resolve_device_precision(args.device, args.precision, args.no_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        logger.info("GPU: %s | device=%s | precision=%s", torch.cuda.get_device_name(device), device, precision)
    else:
        logger.info("Device: CPU | precision=%s", precision)
    model_name = args.model_name or MODEL_CONFIGS[args.model_key]["model_name"]
    logger.info("Model: %s | loss_type: %s | data: %s | output: %s",
                model_name, "mrl", args.data_path, output_dir)
    # Use the existing CSV mapping, two-view collator, and loader without changes.
    loader, tokenizer, _ = build_train_loader(
        args.data_path, model_name, "pair_cls", args.max_length, args.batch_size, args.num_workers)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    if args.gradient_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(encoder.config, "use_cache"):
            encoder.config.use_cache = False
    model = encoder
    dims = resolve_dims(encoder.config.hidden_size, args.matryoshka_dims)
    weights = [1.0] * len(dims) if args.level_weights is None else args.level_weights
    if len(weights) != len(dims) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError(f"level_weights must contain {len(dims)} positive finite weights.")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1) if args.warmup_steps is None else args.warmup_steps
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and the total number of updates.")
    scheduler = get_scheduler("cosine_with_min_lr", optimizer=optimizer,
                              num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                              scheduler_specific_kwargs={"min_lr": args.min_lr})
    amp_enabled = precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    logger.info("Examples: %d | batches/epoch: %d | optimizer steps planned: %d",
                len(loader.dataset), steps_per_epoch, total_steps)
    logger.info("Dims: %s | batch_size: %d | max_length: %d | gradient_checkpointing: %s",
                dims, args.batch_size, args.max_length, args.gradient_checkpointing)

    def encode(batch):
        first, second = batch_inputs(batch, 1, device), batch_inputs(batch, 2, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            q = encoder(**first, return_dict=True).last_hidden_state[:, 0, :]
            p = encoder(**second, return_dict=True).last_hidden_state[:, 0, :]
        return q, p

    configuration = {**vars(args), "model_name": model_name, "matryoshka_dims": dims,
                     "level_weights": weights, "loss_type": "mrl", "pooling": "cls",
                     "warmup_steps": warmup_steps, "scheduler": "cosine_with_min_lr",
                     "loss_reduction": "sum", "singleton_batches": "keep",
                     "resolved_device": str(device), "resolved_precision": precision,
                     "torch_version": str(torch.__version__)}
    (output_dir / "training_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    optimizer_steps = 0
    with (output_dir / "training_metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            model.train()
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            loss_sum, batches, skipped_steps = 0.0, 0, 0
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    q, p = encode(batch)
                    loss, logits = Matry_infonce(q, p, args.temperature, dims, level_weights=weights)
                    loss = loss.float()
                target = torch.arange(q.size(0), device=device)
                metrics = {f"loss_dim_{dim}": torch.nn.functional.cross_entropy(
                    logits[f"dim_{dim}"].detach(), target) for dim in dims}
                metrics["mrl_loss"] = loss.detach()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite encoder loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.max_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                else:
                    grad_norm = torch.stack([p.grad.detach().float().norm() for p in encoder.parameters()
                                             if p.grad is not None]).norm()
                if not scaler.is_enabled() and not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite encoder gradients; try --precision fp32 or a lower learning rate.")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_skipped = scaler.get_scale() < previous_scale
                if not optimizer_skipped:
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                scheduler.step()  # Notebook advances the schedule for every batch.
                values = {key: float(value) for key, value in metrics.items()}
                values.update(loss=float(loss.detach()), epoch=epoch + 1, step=epoch * steps_per_epoch + batches + 1,
                              batch_size=q.size(0),
                              optimizer_step=optimizer_steps, optimizer_skipped=optimizer_skipped,
                              lr=optimizer.param_groups[0]["lr"], grad_scale=scaler.get_scale(),
                              grad_norm=float(grad_norm) if torch.isfinite(grad_norm) else None)
                log.write(json.dumps(values) + "\n")
                loss_sum += values["loss"]
                batches += 1
                if batches % args.log_every == 0:
                    log.flush()
                    logger.info("epoch=%d batch=%d loss=%.6f lr=%.3g optimizer_steps=%d skipped=%d",
                                epoch + 1, batches, values["loss"], values["lr"], optimizer_steps, skipped_steps)
                progress.set_postfix({"loss": f"{values['loss']:.4f}"})
            log.flush()
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Epoch %d complete | mean_loss=%.6f | skipped optimizer steps=%d | saved %s",
                        epoch + 1, loss_sum / max(batches, 1), skipped_steps, output_dir)
            if args.run_eval:
                from MRL.eval import evaluate_model
                evaluate_model(encoder, tokenizer, dims, output_dir, epoch=epoch + 1,
                               eval_work_dir=args.eval_work_dir, batch_size=args.eval_batch_size)
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_name", help="Hugging Face model name or local encoder path")
    source.add_argument("--model_key", choices=list(MODEL_CONFIGS))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/MRL")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=None, help="Default: 10%% of all batches")
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Notebook disables clipping")
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None,
                        help="Prefix dimensions in ascending level weights order; hidden_size is always included")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--level_weights", type=float, nargs="+", default=None,
                        help="Positive weights in ascending dimension order; default is all ones (sum)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cuda:N, or cpu; default selects CUDA when available")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                        help="auto matches notebook CUDA FP16; CPU uses FP32")
    parser.add_argument("--no_amp", action="store_true", help="Compatibility alias for --precision fp32")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Reduce encoder activation memory")
    parser.add_argument("--log_every", type=int, default=10, help="Flush metrics and log progress every N batches")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--run_eval", type=int, choices=[0, 1], default=1, help="Evaluate every epoch")
    parser.add_argument("--eval_work_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--eval_batch_size", type=int, default=64)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
