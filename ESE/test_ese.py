"""Offline checks for PCA targets, layer supervision, and checkpoint evaluation."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer

from ESE.eval import evaluate, evaluation_paths
from ESE.loss import compression_loss, ese_loss, pca_compress, pool_hidden
from ESE.train import parse_args, train


def test_pca_target_and_alignment():
    torch.manual_seed(7)
    embeddings = torch.randn(6, 8, requires_grad=True)
    target = pca_compress(embeddings, 3)
    assert target.shape == (6, 3)
    assert not target.requires_grad
    assert torch.isfinite(target).all()
    # Full-rank compression must preserve the feature-dependency projection's
    # Gram matrix, independent of SVD signs/rotations.
    full_target = pca_compress(embeddings, 8)
    dependency = (embeddings.detach().T @ embeddings.detach() / math.sqrt(8)).softmax(-1)
    projected = embeddings.detach() @ dependency
    torch.testing.assert_close(full_target @ full_target.T, projected @ projected.T, atol=1e-5, rtol=1e-5)
    assert compression_loss(target, target, 1.0).item() == pytest.approx(0, abs=1e-6)
    prefix = embeddings[:, :3]
    compression_loss(prefix, target, 1.0).backward()
    assert embeddings.grad[:, :3].abs().sum() > 0
    assert embeddings.grad[:, 3:].abs().sum() == 0


def test_pooling_ignores_padding():
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(pool_hidden(hidden, mask, "mean"), torch.tensor([[3.0, 6.0]]))
    torch.testing.assert_close(pool_hidden(hidden, mask, "cls"), hidden[:, 0])


def test_all_blocks_receive_supervision_without_embedding_loss():
    torch.manual_seed(11)
    first = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(4)]
    second = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(4)]
    mask = torch.ones(4, 3, dtype=torch.long)
    loss, metrics = ese_loss(first, second, mask, mask, compression_size=3)
    assert torch.isfinite(loss)
    assert metrics["compress_loss"] > 0
    loss.backward()
    assert first[0].grad is None and second[0].grad is None
    for states in (first, second):
        assert all(state.grad is not None and torch.isfinite(state.grad).all()
                   and state.grad.abs().sum() > 0 for state in states[1:])
    # Shallower blocks receive larger weights; the embedding state is excluded.
    task_losses = []
    for left, right in zip(first[1:], second[1:]):
        a, b = left.mean(1)[:, :3], right.mean(1)[:, :3]
        logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / 0.07
        task_losses.append(F.cross_entropy(logits, torch.arange(4)))
    expected = task_losses[-1] + (task_losses[0] + task_losses[1] / (1 + math.log(2))) / 2
    torch.testing.assert_close(metrics["express_loss"], expected)


def test_single_block_and_invalid_compression():
    states = [torch.randn(2, 2, 4, requires_grad=True) for _ in range(2)]
    mask = torch.ones(2, 2, dtype=torch.long)
    loss, _ = ese_loss(states, states, mask, mask, compression_size=4)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="compression_size"):
        ese_loss(states, states, mask, mask, compression_size=5)


def test_offline_train_save_reload_and_evaluate(tmp_path):
    # Keep CPU smoke checks small and independent of model downloads/GPU access.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model_dir = tmp_path / "encoder"
        model_dir.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "happy", "sad",
                 "day", "night", "sun", "rain"]
        (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizer(str(model_dir / "vocab.txt"))
        assert tokenizer("good day")["input_ids"] != tokenizer("bad night")["input_ids"]
        tokenizer.save_pretrained(model_dir)
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=3,
                            num_attention_heads=2, intermediate_size=32, max_position_embeddings=512)
        torch.manual_seed(3)
        original = BertModel(config)
        original.save_pretrained(model_dir)
        texts = ["good day", "bad night", "happy sun", "sad rain", "good sun", "bad rain",
                 "happy day", "sad night"]
        training_csv = tmp_path / "train.csv"
        pd.DataFrame({"text": texts}).to_csv(training_csv, index=False)
        eval_work_dir = tmp_path / "ESE"
        tasks = evaluation_paths(eval_work_dir)
        for paths in tasks["classification"]:
            for path in paths:
                destination = Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"text": texts, "label": [0, 1] * 4}).to_csv(path, index=False)
        for kind, label_name in (("pair", "label"), ("sts", "score")):
            for path in tasks[kind]:
                destination = Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"sentence1": texts, "sentence2": texts[::-1],
                              label_name: ([0, 1] * 4 if kind == "pair" else list(range(8)))
                              }).to_csv(path, index=False)
        output_dir = tmp_path / "checkpoint"
        args = parse_args([
            "--model_name", str(model_dir), "--data_path", str(training_csv),
            "--output_dir", str(output_dir), "--device", "cpu", "--epochs", "1",
            "--batch_size", "4", "--num_workers", "0", "--max_length", "16",
            "--ese_compression_size", "4", "--matryoshka_dims", "4", "16",
            "--gradient_checkpointing", "--run_eval", "1", "--eval_work_dir", str(eval_work_dir),
        ])
        trained = train(args)
        assert not torch.equal(original.embeddings.word_embeddings.weight,
                               trained.embeddings.word_embeddings.weight)
        assert (output_dir / "model.safetensors").is_file()
        assert (output_dir / "tokenizer_config.json").is_file()
        saved_config = json.loads((output_dir / "training_config.json").read_text())
        assert saved_config["pooling"] == "mean"
        assert saved_config["training_dims"] == [4, 16]
        rows = [json.loads(row) for row in (output_dir / "training_metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and rows[-1]["optimizer_step"] == 2
        assert all(math.isfinite(row["loss"]) and row["compress_loss"] > 0 for row in rows)
        per_epoch = json.loads((output_dir / "result_epoch_1.json").read_text())
        assert len(per_epoch["classification"]) == 3
        assert len(per_epoch["pair"]) == 3
        assert len(per_epoch["sts"]) == 8
        reloaded = evaluate(output_dir, device="cpu", eval_work_dir=eval_work_dir, batch_size=4)
        for name in ("classification", "pair", "sts"):
            assert reloaded[name] == per_epoch[name]
    finally:
        torch.set_num_threads(previous_threads)
