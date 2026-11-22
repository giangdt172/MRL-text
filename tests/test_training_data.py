"""Offline regression checks for shared positive pairs and all four trainers."""

import importlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from common.data_preparation import DEFAULT_TRAIN_DATA_PATH, PositivePairDataset, build_train_loader, load_training_pairs
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


def test_default_pairs_are_local_and_do_not_download(monkeypatch, pairs_csv):
    monkeypatch.setattr("common.data_preparation.DEFAULT_TRAIN_DATA_PATH", pairs_csv)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Training must read the local CSV without contacting the Hub.")

    monkeypatch.setattr("datasets.load_dataset", unexpected_download)
    pairs = load_training_pairs()
    assert len(pairs) == 7 and not pairs.metadata["oversampled"]
    assert pairs[0] == ("good", "good day sun")
    assert pairs.metadata["path"] == str(pairs_csv.resolve())
    smaller = load_training_pairs(max_train_samples=2, seed=9)
    repeat = load_training_pairs(max_train_samples=2, seed=9)
    assert len(smaller) == 2 and smaller.pairs == repeat.pairs


def test_missing_local_pairs_explain_download_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="bash download.sh"):
        load_training_pairs(tmp_path / "missing.csv")


def test_download_script_writes_inside_project_from_any_directory(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "download.sh"
    script.write_text((Path(__file__).resolve().parent.parent / "download.sh").read_text())
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "datasets.py").write_text('''import csv
import os
from pathlib import Path

def load_dataset(name, config, split):
    assert (name, config, split) == ("sentence-transformers/quora-duplicates", "pair", "train")
    assert Path(os.environ["HF_HOME"]) == Path.cwd() / "data" / ".cache" / "huggingface"
    assert Path(os.environ["HF_DATASETS_CACHE"]) == Path(os.environ["HF_HOME"]) / "datasets"
    assert Path(os.environ["HF_HUB_CACHE"]) == Path(os.environ["HF_HOME"]) / "hub"
    return LocalDataset()

class LocalDataset:
    def to_csv(self, path, index):
        assert index is False
        with open(path, "w", newline="") as stream:
            csv.writer(stream).writerows([("anchor", "positive"), ("a", "A"), ("b", "B")])
''')
    subprocess.run(["bash", str(script)], cwd=tmp_path, check=True,
                   env={**os.environ, "PYTHONPATH": str(stub), "HF_DATASETS_CACHE": "/tmp/outside-datasets",
                        "HF_HUB_CACHE": "/tmp/outside-hub"})
    pairs = load_training_pairs(project / "data" / "quora-duplicates-pair.csv")
    assert pairs.pairs == [("a", "A"), ("b", "B")]
    assert not (tmp_path / "data").exists()


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
def test_each_cli_defaults_to_project_pairs(method, pairs_csv):
    module = importlib.import_module(f"{method}.train")
    args = module.parse_args(["--model_key", "bert", "--run_eval", "0"])
    assert Path(args.data_path) == DEFAULT_TRAIN_DATA_PATH
    args.data_path = str(pairs_csv)
    module.validate_args(args)
    assert args.data_path == str(pairs_csv.resolve()) and args.max_train_samples == 150_000
