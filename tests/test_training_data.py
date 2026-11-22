"""Offline regression checks for shared positive pairs and all four trainers."""

import importlib
import json

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from common.data_preparation import PositivePairDataset, build_train_loader, load_training_pairs
from common.loss import info_nce


def test_pairs_stay_aligned_and_batches_are_reproducible(encoder_dir, pairs_csv):
    def make_loader():
        return build_train_loader(pairs_csv, str(encoder_dir), batch_size=3, num_workers=0, seed=17)[0]

    left = make_loader()
    torch.manual_seed(987)  # Model RNG must not change the data order.
    right = make_loader()
    left_batches, right_batches = list(left), list(right)
    assert len(left.dataset) == 7 and len(left_batches) == 3
    assert left_batches[-1]["anchor"]["input_ids"].size(0) == 1
    for a, b in zip(left_batches, right_batches):
        for side in ("anchor", "positive"):
            assert "token_type_ids" in a[side]
            torch.testing.assert_close(a[side]["input_ids"], b[side]["input_ids"])
        assert a["anchor"]["input_ids"].shape[1] < a["positive"]["input_ids"].shape[1]
    assert left.dataset[0] == ("good", "good day sun")
    assert left.dataset.metadata["pairs_sha256"] == right.dataset.metadata["pairs_sha256"]


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"text": ["a", "b"]}),
    pd.DataFrame({"anchor": ["a", None], "positive": ["b", "c"]}),
    pd.DataFrame({"anchor": ["a", "b"], "positive": ["c", "  "]}),
    pd.DataFrame({"anchor": ["a"], "positive": ["b"]}),
])
def test_reject_invalid_or_single_text_data(frame):
    with pytest.raises(ValueError):
        PositivePairDataset(frame)


def test_hub_uses_only_pair_train_without_oversampling(monkeypatch):
    from datasets import Dataset

    raw = Dataset.from_dict({"anchor": ["a", "b", "c"], "positive": ["A", "B", "C"]})
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setattr("datasets.load_dataset", load)
    pairs = load_training_pairs(data_cache_dir="/tmp/quora-test-cache", dataset_revision="test-commit")
    assert calls == [(("sentence-transformers/quora-duplicates", "pair"),
                      {"split": "train", "cache_dir": "/tmp/quora-test-cache", "revision": "test-commit"})]
    assert len(pairs) == 3 and not pairs.metadata["oversampled"]
    assert pairs[0] == ("a", "A")
    smaller = PositivePairDataset(raw.to_pandas(), max_train_samples=2, seed=9)
    repeat = PositivePairDataset(raw.to_pandas(), max_train_samples=2, seed=9)
    assert len(smaller) == 2 and smaller.pairs == repeat.pairs


def test_common_infonce_preserves_losses_and_gradients():
    from ESE.loss import info_nce as ese_nce
    from FlagMRL.encoder_loss import encoder_losses
    from MIPIC.loss import Matry_infonce as mipic_nce
    from MRL.loss import Matry_infonce as mrl_nce

    torch.manual_seed(5)
    q = torch.randn(4, 8, requires_grad=True)
    p = torch.randn(4, 8, requires_grad=True)
    target = torch.arange(4)
    expected = sum(F.cross_entropy(F.normalize(q[:, :d], dim=-1) @
                                  F.normalize(p[:, :d], dim=-1).T / 0.2, target) for d in (4, 8))
    for actual in (mrl_nce(q, p, 0.2, [4, 8])[0], mipic_nce(q, p, 0.2, [4, 8])[0],
                   encoder_losses(q, p, target, [4, 8], 0.2)[0]):
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, (q, p), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (q, p), retain_graph=True)
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(ese_nce(q, p, 0.2), info_nce(q, p, 0.2)[0])
    zeros = torch.zeros(3, 4, dtype=torch.float16, requires_grad=True)
    loss, _ = info_nce(zeros, zeros)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(zeros.grad).all()


@pytest.mark.parametrize("method", ["MRL", "MIPIC", "FlagMRL", "ESE"])
def test_all_trainers_use_pairs_and_save_checkpoints(method, encoder_dir, pairs_csv, tmp_path):
    module = importlib.import_module(f"{method}.train")
    output_dir = tmp_path / method
    arguments = ["--model_name", str(encoder_dir), "--data_path", str(pairs_csv),
                 "--output_dir", str(output_dir), "--device", "cpu", "--epochs", "1",
                 "--batch_size", "3", "--num_workers", "0", "--max_length", "16",
                 "--matryoshka_dims", "4", "8", "16", "--run_eval", "0"]
    if method == "MIPIC":
        arguments += ["--align_layers", "1", "2", "--pipeline_pairs", "1", "4", "2", "8", "3", "16",
                      "--base_k", "2", "--d_att", "2"]
    if method == "ESE":
        arguments += ["--ese_compression_size", "4"]
    args = module.parse_args(arguments)
    module.train(args)
    config = json.loads((output_dir / "training_config.json").read_text())
    assert config["training_data"]["pairs_sha256"] == load_training_pairs(pairs_csv).metadata["pairs_sha256"]
    assert config["training_data"]["num_pairs"] == 7
    weights_dir = output_dir / "encoder" if method == "FlagMRL" else output_dir
    assert (weights_dir / "model.safetensors").is_file()
    rows = [json.loads(row) for row in (output_dir / "training_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == (2 if method == "FlagMRL" else 3)
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in rows)


@pytest.mark.parametrize("method", ["MRL", "MIPIC", "FlagMRL", "ESE"])
def test_each_cli_defaults_to_hub_pairs(method):
    module = importlib.import_module(f"{method}.train")
    args = module.parse_args(["--model_key", "bert", "--run_eval", "0"])
    module.validate_args(args)
    assert args.data_path is None and args.max_train_samples == 150_000
