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
from transformers import AutoModel

from MIPIC.config import MODEL_CONFIGS, model_name_map_short_name
from MIPIC.data_preparation import build_train_loader
from MIPIC.loss import (
    CKALoss, HorizontalAttentionAlignment, SubmatrixCKALoss,
    PipelineInfoNCELoss, TotalAlignmentLoss, Matry_infonce, pool_hidden,
)


def batch_inputs(batch, side, device):
    return {key: batch[f"{key}{side}_stu"].to(device, non_blocking=device.type == "cuda")
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if f"{key}{side}_stu" in batch}

def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("epochs, batch_size and max_length must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be positive and finite.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite.")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    if args.log_every < 1:
        raise ValueError("log_every must be positive.")
    if args.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive.")
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
    logger = logging.getLogger("MIPIC.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log", mode="w")):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def resolve_alignment(model_name, encoder_config, args):
    """Keep original schedules for known encoders; derive one for local models."""
    full_dim, depth = encoder_config.hidden_size, encoder_config.num_hidden_layers
    if depth < 2 or full_dim < 2:
        raise ValueError("MIPIC requires an encoder with at least two layers and two hidden dimensions.")
    key = args.model_key or model_name_map_short_name.get(model_name)
    known = MODEL_CONFIGS.get(key, {})
    if known and known["d_full"] != full_dim:
        raise ValueError(f"Encoder hidden_size={full_dim} does not match {key} schedule.")
    dims = args.matryoshka_dims
    if dims is None:
        dims = known.get("matryoshka_dims") or [d for d in (16, 32, 64, 128, 256, 512) if d < full_dim] + [full_dim]
        if len(dims) == 1:
            dims = [max(1, full_dim // 2), full_dim]
    if not dims or any(d < 1 or d > full_dim for d in dims):
        raise ValueError("matryoshka_dims must be in [1, hidden_size].")
    dims = sorted(set(dims + [full_dim]))
    if len(dims) < 2:
        raise ValueError("MIPIC needs at least one prefix dimension smaller than hidden_size.")
    layers = args.align_layers or known.get("align_layers")
    if layers is None:
        layers = sorted(set((max(1, depth // 3), max(1, 2 * depth // 3))))
    if any(layer < 1 or layer > depth for layer in layers):
        raise ValueError(f"Alignment layers must be in [1, {depth}] (hidden_states indices).")
    pipes = [args.pipeline_pairs] if args.pipeline_pairs is not None else known.get("pipeline_pairs")
    if pipes is None:
        # At most one stage per encoder layer, spanning narrow/shallow to full/deep.
        count = min(len(dims), depth)
        pipe_dims = [dims[round(i * (len(dims) - 1) / (count - 1))] for i in range(count)]
        pipe_layers = [1 + round(i * (depth - 1) / (count - 1)) for i in range(count)]
        pipes = [sum(([layer, dim] for layer, dim in zip(pipe_layers, pipe_dims)), [])]
    for pipe in pipes:
        if len(pipe) < 4 or len(pipe) % 2 or any(layer < 1 or layer > depth for layer in pipe[::2]):
            raise ValueError(f"Pipeline needs (layer, dim) stages with layer in [1, {depth}].")
    return dims, list(layers), [list(pipe) for pipe in pipes]


def train(args):
    validate_args(args)
    if args.run_eval:
        from MIPIC.eval import check_eval_data
        check_eval_data(args.eval_work_dir)
    if args.num_workers < 0 or args.base_k < 1 or args.d_att < 1:
        raise ValueError("num_workers must be nonnegative; base_k/d_att must be positive.")
    device, precision = resolve_device_precision(args.device, args.precision, args.no_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model_name = args.model_name or MODEL_CONFIGS[args.model_key]["model_name"]
    logger.info("Model: %s | device: %s | precision: %s | output: %s", model_name, device, precision, output_dir)
    loader, tokenizer, _ = build_train_loader(
        args.data_path, model_name, "pair_cls", args.max_length, args.batch_size, args.num_workers)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    if args.gradient_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False
    dims, layers, pipes = resolve_alignment(model_name, encoder.config, args)
    alignment = TotalAlignmentLoss(
        d_full=encoder.config.hidden_size, matryoshka_dims=dims, align_layers=layers,
        pipeline_pairs=pipes, alpha=0.4, beta=0.4, gamma=0.2,
        base_k=args.base_k, d_att=args.d_att).to(device)
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr},
        {"params": alignment.parameters(), "lr": args.lr * 2},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = len(loader)
    # The notebook reassigns optimizer after constructing its scheduler, leaving
    # the actual training optimizer at constant LR. Reproduce that effect
    # directly, without creating an unused optimizer/scheduler or its warnings.
    if args.warmup_steps not in {None, 0}:
        raise ValueError("Notebook MIPIC training uses constant LR; warmup_steps must be 0.")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    configuration = {
        **vars(args), "model_name": model_name, "matryoshka_dims": dims,
        "align_layers": layers, "pipeline_pairs": pipes, "warmup_steps": 0,
        "scheduler": "constant_notebook_effective", "alignment_lr": args.lr * 2,
        "attention_kl_enabled": False, "cka_dtype": "float64", "selected_padding": "keep",
        "gradient_clipping": "separate_encoder_alignment", "singleton_batches": "keep",
        "loss_type": "mipic", "resolved_device": str(device), "resolved_precision": precision,
        "torch_version": str(torch.__version__),
    }
    (output_dir / "training_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    logger.info("Examples: %d | updates/epoch: %d | dims: %s | alignment layers: %s | pipeline: %s",
                len(loader.dataset), steps_per_epoch, dims, layers, pipes)
    optimizer_steps = 0
    with (output_dir / "training_metrics.jsonl").open("w") as metrics_log:
        for epoch in range(args.epochs):
            encoder.train()
            alignment.train()
            loss_sum, batches, skipped_steps = 0.0, 0, 0
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                first, second = batch_inputs(batch, 1, device), batch_inputs(batch, 2, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                    out1 = encoder(**first, output_hidden_states=True, return_dict=True)
                    out2 = encoder(**second, output_hidden_states=True, return_dict=True)
                    losses1 = alignment(out1.hidden_states, first["attention_mask"])
                    losses2 = alignment(out2.hidden_states, second["attention_mask"])
                    losses = {key: (losses1[key] + losses2[key]) / 2 for key in losses1}
                    q = pool_hidden(out1.last_hidden_state, first["attention_mask"], args.pooling)
                    p = pool_hidden(out2.last_hidden_state, second["attention_mask"], args.pooling)
                    contrastive, _ = Matry_infonce(q, p, args.temperature, dims)
                    loss = (0.4 * losses["total_loss"] + 0.6 * contrastive).float()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite MIPIC loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                alignment_grad_norm = torch.nn.utils.clip_grad_norm_(alignment.parameters(), args.max_grad_norm)
                grad_norm = torch.hypot(encoder_grad_norm, alignment_grad_norm)
                if not scaler.is_enabled() and not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite gradients; try fp32 or a lower learning rate.")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_skipped = scaler.get_scale() < previous_scale
                if not optimizer_skipped:
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                values = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                          for key, value in losses.items()}
                values.update(loss=float(loss.detach()), contrastive_loss=float(contrastive.detach()),
                              epoch=epoch + 1, step=epoch * steps_per_epoch + batches + 1,
                              optimizer_step=optimizer_steps, optimizer_skipped=optimizer_skipped,
                              lr=optimizer.param_groups[0]["lr"], alignment_lr=optimizer.param_groups[1]["lr"],
                              batch_size=first["input_ids"].size(0), grad_scale=scaler.get_scale(),
                              grad_norm=float(grad_norm) if torch.isfinite(grad_norm) else None)
                metrics_log.write(json.dumps(values, allow_nan=False) + "\n")
                loss_sum += values["loss"]
                batches += 1
                if batches % args.log_every == 0:
                    metrics_log.flush()
                    logger.info("epoch=%d batch=%d loss=%.6f att=%.6f cka=%.6f chain=%.6f updates=%d skipped=%d",
                                epoch + 1, batches, values["loss"], values["att_loss"],
                                values["cka_loss"], values["chain_loss"], optimizer_steps, skipped_steps)
                progress.set_postfix(loss=f"{values['loss']:.4f}")
            metrics_log.flush()
            encoder.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(alignment.state_dict(), output_dir / "alignment_loss.pt")
            logger.info("Epoch %d complete | mean_loss=%.6f | skipped updates=%d | saved %s",
                        epoch + 1, loss_sum / max(batches, 1), skipped_steps, output_dir)
            if args.run_eval:
                from MIPIC.eval import evaluate_model
                evaluate_model(encoder, tokenizer, dims, output_dir, epoch=epoch + 1,
                               eval_work_dir=args.eval_work_dir, batch_size=args.eval_batch_size)
    return encoder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_name", help="Hugging Face model name or local encoder path")
    source.add_argument("--model_key", choices=list(MODEL_CONFIGS))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/MIPIC")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Must be 0: notebook effective LR is constant")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=None)
    parser.add_argument("--align_layers", type=int, nargs="+", default=None)
    parser.add_argument("--pipeline_pairs", type=int, nargs="+", default=None,
                        help="Flat chain: layer1 dim1 layer2 dim2 ... (hidden_states indices)")
    parser.add_argument("--base_k", type=int, default=64)
    parser.add_argument("--d_att", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--pooling", choices=["cls"], default="cls",
                        help="Notebook takes hidden[:, 0, :] including left padding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--run_eval", type=int, choices=[0, 1], default=1, help="Evaluate every epoch")
    parser.add_argument("--eval_work_dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--eval_batch_size", type=int, default=64)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())


