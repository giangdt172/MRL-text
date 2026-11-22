"""Prefix/pooling regressions and real MTEB runs on a tiny offline STS fixture."""

import importlib
import json
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from common.mteb_eval import MatryoshkaEncoder, evaluate, evaluate_model
from common.mteb_tasks import DEFAULT_EVAL_TASKS, EVAL_TASK_GROUPS
from FlagMRL import FlagProjector
from FlagMRL.model import FlagTextModel


@pytest.mark.parametrize("pooling", ["cls", "mean"])
@pytest.mark.parametrize("flag", [False, True])
def test_adapter_pool_project_truncate_normalize(encoder_dir, pooling, flag):
    encoder = AutoModel.from_pretrained(encoder_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    model = encoder
    if flag:
        projector = FlagProjector(16, [4, 16], init="random", seed=7)
        model = FlagTextModel(encoder, projector).eval()
    texts = ["good", "bad night rain", "good day"]
    features = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder(**features).last_hidden_state
        if flag:
            hidden = hidden @ projector.U
        if pooling == "cls":
            expected = hidden[:, 0]
        else:
            mask = features["attention_mask"].unsqueeze(-1)
            expected = (hidden * mask).sum(1) / mask.sum(1)
        expected = F.normalize(expected[:, :4], dim=-1).numpy()
    adapter = MatryoshkaEncoder(model, tokenizer, 4, pooling=pooling, batch_size=2)
    actual = adapter.encode([{"text": texts}])
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
    assert adapter.encode([]).shape == (0, 4)
    assert adapter.similarity(actual, actual).shape == (3, 3)
    torch.testing.assert_close(adapter.similarity_pairwise(actual, actual), torch.ones(3))


@pytest.fixture
def offline_mteb(monkeypatch, tmp_path):
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    mteb = pytest.importorskip("mteb")
    from datasets import Dataset

    original_get_task = mteb.get_task
    requests = []

    def get_task(name, **kwargs):
        requests.append((name, kwargs))
        task = original_get_task(name, **kwargs)
        task.metadata = task.metadata.model_copy(update={"superseded_by": None})
        task.dataset = {"test": Dataset.from_dict({
            "sentence1": ["good day", "bad night", "sun", "rain", "good", "day sun"],
            "sentence2": ["good day", "good day", "sun day", "bad night", "rain", "sun"],
            "score": [5.0, 0.0, 4.0, 2.0, 1.0, 3.0],
        })}
        task.data_loaded = True
        return task

    monkeypatch.setattr(mteb, "get_task", get_task)
    return mteb, requests


def test_real_mteb_selected_tasks_results_and_rng_restoration(encoder_dir, offline_mteb, tmp_path):
    _, requests = offline_mteb
    model = AutoModel.from_pretrained(encoder_dir).train()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    python_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = evaluate_model(model, tokenizer, [4, 16], tmp_path, eval_tasks=["STSBenchmark"],
                            epoch=1, pooling="mean", batch_size=2)
    assert model.training
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    assert requests == [("STSBenchmark", {"eval_splits": ["test"], "languages": ["eng"]})] * 2
    assert result["tasks"] == ["STSBenchmark"]
    assert set(result["scores"]) == {"dim_4", "dim_16"}
    assert all(np.isfinite(scores["STSBenchmark"]) for scores in result["scores"].values())
    for path in result["raw_results"].values():
        assert (tmp_path / path).is_file()
    assert json.loads((tmp_path / "result_epoch_1.json").read_text()) == result
    assert (tmp_path / "summary_epoch_1.txt").is_file()


def test_mteb_failure_restores_training_mode_and_rng(encoder_dir, offline_mteb, monkeypatch, tmp_path):
    mteb, _ = offline_mteb
    model = AutoModel.from_pretrained(encoder_dir).train()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    state = torch.get_rng_state()

    def fail(*args, **kwargs):
        torch.manual_seed(123)
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(mteb, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate_model(model, tokenizer, [4], tmp_path, eval_tasks=["STSBenchmark"])
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("method", ["MRL", "MIPIC", "FlagMRL", "ESE"])
def test_training_epoch_uses_mteb_and_checkpoint_reloads(method, encoder_dir, pairs_csv, offline_mteb, tmp_path):
    module = importlib.import_module(f"{method}.train")
    output = tmp_path / method
    argv = ["--model_name", str(encoder_dir), "--data_path", str(pairs_csv), "--output_dir", str(output),
            "--device", "cpu", "--epochs", "1", "--batch_size", "4", "--num_workers", "0",
            "--matryoshka_dims", "4", "16", "--run_eval", "1", "--eval_tasks", "STSBenchmark",
            "--eval_batch_size", "3"]
    if method == "MIPIC":
        argv += ["--align_layers", "1", "2", "--pipeline_pairs", "1", "4", "2", "16", "--base_k", "2", "--d_att", "2"]
    if method == "ESE":
        argv += ["--ese_compression_size", "4"]
    model = module.train(module.parse_args(argv))
    assert model.training
    saved = json.loads((output / "result_epoch_1.json").read_text())
    reloaded = evaluate(output, method=method.lower(), eval_tasks=["STSBenchmark"], device="cpu",
                        eval_batch_size=2, output_dir=tmp_path / "reloaded")
    assert reloaded["scores"] == saved["scores"]


@pytest.mark.parametrize("method", ["MRL", "MIPIC", "FlagMRL", "ESE"])
def test_default_eval_uses_only_selected_preset(method):
    module = importlib.import_module(f"{method}.train")
    args = module.parse_args(["--model_key", "bert"])
    assert args.run_eval == 1 and tuple(args.eval_tasks) == DEFAULT_EVAL_TASKS
    module.validate_args(args)
    override = module.parse_args(["--model_key", "bert", "--eval_tasks", "STSBenchmark", "SICK-R"])
    assert override.eval_tasks == ["STSBenchmark", "SICK-R"]
    args.eval_tasks = []
    with pytest.raises(ValueError, match="eval_tasks"):
        module.validate_args(args)


def test_selected_preset_resolves_exact_task_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    mteb = pytest.importorskip("mteb")
    assert len(DEFAULT_EVAL_TASKS) == len(set(DEFAULT_EVAL_TASKS)) == 21
    for task_type, names in EVAL_TASK_GROUPS.items():
        for name in names:
            task = mteb.get_task(name, eval_splits=["test"], languages=["eng"])
            assert task.metadata.name == name
            assert task.metadata.type == task_type
            assert "test" in task.metadata.eval_splits and task.hf_subsets
