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


def test_adapter_reuses_full_embeddings_and_restores_duplicate_rows(encoder_dir):
    model = AutoModel.from_pretrained(encoder_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    texts = ["good", "bad night", "good", "day sun"]
    with torch.no_grad():
        features = tokenizer(texts, padding=True, return_tensors="pt")
        full = model(**features).last_hidden_state[:, 0].float()
    encoded_rows = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: encoded_rows.append(kwargs["input_ids"].size(0)), with_kwargs=True)
    try:
        adapter = MatryoshkaEncoder(model, tokenizer, 4, batch_size=2)
        small = adapter.encode([{"text": texts}])
        assert sum(encoded_rows) == 3  # Three unique strings, not four rows.
        adapter.set_dim(16)
        # Changed batch boundaries must still reuse the same full embeddings.
        large = adapter.encode([{"text": texts[:1]}, {"text": texts[1:]}])
        assert sum(encoded_rows) == 3
        np.testing.assert_allclose(small, F.normalize(full[:, :4], dim=-1).numpy(), atol=1e-6)
        np.testing.assert_allclose(large, F.normalize(full, dim=-1).numpy(), atol=1e-6)
        np.testing.assert_array_equal(large[0], large[2])
        adapter.clear_cache()
        assert not adapter._embedding_cache
        adapter.encode([{"text": texts}])
        assert sum(encoded_rows) == 6
    finally:
        hook.remove()


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
    assert requests == [("STSBenchmark", {"eval_splits": ["test"], "languages": ["eng"]})]
    assert result["tasks"] == ["STSBenchmark"]
    assert set(result["scores"]) == {"dim_4", "dim_16"}
    assert all(np.isfinite(scores["STSBenchmark"]) for scores in result["scores"].values())
    for path in result["raw_results"].values():
        assert (tmp_path / path).is_file()
    assert json.loads((tmp_path / "result_epoch_1.json").read_text()) == result
    assert (tmp_path / "summary_epoch_1.txt").is_file()
    assert result["metrics"] == {"STSBenchmark": "cosine_spearman"}
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "results_long.csv").is_file()
    assert "| 1 | cosine_spearman |" in (tmp_path / "summary_all_epochs.txt").read_text()


def test_tasks_run_through_all_dims_with_separate_caches(encoder_dir, offline_mteb, monkeypatch, tmp_path):
    _, requests = offline_mteb
    model = AutoModel.from_pretrained(encoder_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    encoded_rows, cleared = [], []
    original_clear = MatryoshkaEncoder.clear_cache

    def clear(adapter):
        cleared.append(len(adapter._embedding_cache))
        original_clear(adapter)

    monkeypatch.setattr(MatryoshkaEncoder, "clear_cache", clear)
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: encoded_rows.append(kwargs["input_ids"].size(0)), with_kwargs=True)
    try:
        result = evaluate_model(model, tokenizer, [4, 16], tmp_path,
                                eval_tasks=["STSBenchmark", "SICK-R"], batch_size=2)
    finally:
        hook.remove()
    assert [name for name, kwargs in requests] == ["STSBenchmark", "SICK-R"]
    assert sum(encoded_rows) == 14  # Seven unique strings per task, independent of dimension count.
    assert cleared == [7, 7]
    for key, path in result["raw_results"].items():
        raw = json.loads((tmp_path / path).read_text())
        assert [task["task_name"] for task in raw["task_results"]] == ["STSBenchmark", "SICK-R"]
        assert set(result["scores"][key]) == {"STSBenchmark", "SICK-R"}


@pytest.mark.parametrize("name", ["Banking77Classification", "ArguAna", "TwentyNewsgroupsClustering.v2"])
def test_cached_dims_match_independent_mteb_runs(name, encoder_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    mteb = pytest.importorskip("mteb")
    from datasets import Dataset

    original_get_task = mteb.get_task

    def get_task(task_name, **kwargs):
        task = original_get_task(task_name, **kwargs)
        task.metadata = task.metadata.model_copy(update={"superseded_by": None})
        if task_name == "Banking77Classification":
            task.n_experiments = 2
            task.samples_per_label = 2
            task.dataset = {
                "train": Dataset.from_dict({"text": ["good", "good day", "sun", "good sun",
                                                       "bad", "bad night", "rain", "bad rain"],
                                            "label": [0] * 4 + [1] * 4}),
                "test": Dataset.from_dict({"text": ["good day", "sun", "bad night", "rain"],
                                           "label": [0, 0, 1, 1]}),
            }
        elif task_name == "TwentyNewsgroupsClustering.v2":
            task.n_clusters = 2
            task.max_fraction_of_documents_to_embed = None
            task.max_document_to_embed = 6
            task.max_documents_per_cluster = 4
            task.dataset = {"test": Dataset.from_dict({
                "sentences": ["good", "good day", "sun", "good sun", "bad", "bad night", "rain", "bad rain"],
                "labels": [["a"]] * 4 + [["b"]] * 4,
            })}
        else:
            task.dataset = {"default": {"test": {
                "corpus": Dataset.from_dict({"id": ["d1", "d2", "d3", "d4"],
                                             "text": ["good day", "bad night", "sun", "rain"]}),
                "queries": Dataset.from_dict({"id": ["q1", "q2"], "text": ["good", "bad"]}),
                "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1}}, "top_ranked": None,
            }}}
        task.data_loaded = True
        return task

    monkeypatch.setattr(mteb, "get_task", get_task)
    model = AutoModel.from_pretrained(encoder_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    cached = evaluate_model(model, tokenizer, [4, 16], tmp_path / "cached", eval_tasks=[name], batch_size=2)
    for dim in (4, 16):
        independent = evaluate_model(model, tokenizer, [dim], tmp_path / f"single-{dim}",
                                     eval_tasks=[name], batch_size=2)
        assert cached["scores"][f"dim_{dim}"] == independent["scores"][f"dim_{dim}"]


def test_mteb_loads_uncached_json_dataset(encoder_dir, monkeypatch, tmp_path):
    """Exercise MTEB -> load_dataset -> DownloadManager, including first-time loading."""
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    mteb = pytest.importorskip("mteb")
    from mteb.abstasks import abstask

    original_get_task = mteb.get_task
    original_load = abstask.load_dataset
    original_evaluate = mteb.evaluate
    loads, transforms, evaluated, sampled = [], [], [], []

    def load_dataset(**kwargs):
        loads.append(kwargs)
        return original_load(**kwargs)

    def transform(task, **kwargs):
        transforms.append(task)
        # Loading may change task attributes and consume global and task-local RNGs.
        task.loaded_marker = task.rng_state.random()
        random.random()
        np.random.random()
        torch.rand(1)
        task.np_rng.random()

    def run(adapter, *, tasks, **kwargs):
        task = tasks[0]
        assert task.data_loaded and task.loaded_marker == transforms[0].loaded_marker
        evaluated.append(task)
        assert task.dataset is transforms[0].dataset
        sampled.append((random.random(), np.random.random(), torch.rand(1).item(),
                        task.rng_state.random(), task.np_rng.random()))
        return original_evaluate(adapter, tasks=tasks, **kwargs)

    monkeypatch.setattr(abstask, "load_dataset", load_dataset)
    monkeypatch.setattr(mteb, "evaluate", run)
    data_file = tmp_path / "sts.jsonl"
    records = [
        {"sentence1": "good day", "sentence2": "good day", "score": 5.0},
        {"sentence1": "bad night", "sentence2": "good day", "score": 0.0},
        {"sentence1": "sun", "sentence2": "sun day", "score": 4.0},
        {"sentence1": "rain", "sentence2": "bad night", "score": 2.0},
        {"sentence1": "good", "sentence2": "rain", "score": 1.0},
        {"sentence1": "day sun", "sentence2": "sun", "score": 3.0},
    ]
    data_file.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    def get_task(name, **kwargs):
        task = original_get_task(name, **kwargs)
        monkeypatch.setattr(type(task), "dataset_transform", transform)
        task.metadata = task.metadata.model_copy(update={
            "superseded_by": None,
            "dataset": {"path": "json", "revision": "local-test", "data_files": {"test": str(data_file)},
                        "cache_dir": str(tmp_path / "dataset-cache")},
        })
        assert not task.data_loaded
        return task

    monkeypatch.setattr(mteb, "get_task", get_task)
    model = AutoModel.from_pretrained(encoder_dir).train()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    result = evaluate_model(model, tokenizer, [4, 16], tmp_path / "results",
                            eval_tasks=["STSBenchmark"], batch_size=2)
    assert model.training
    assert set(result["scores"]) == {"dim_4", "dim_16"}
    assert all(np.isfinite(scores["STSBenchmark"]) for scores in result["scores"].values())
    assert list((tmp_path / "dataset-cache").rglob("*.arrow"))
    assert len(loads) == len(transforms) == 1
    assert len(evaluated) == 2 and evaluated[0] is not evaluated[1]
    assert sampled[0] == sampled[1]
    assert result["dataset_load_seconds"]["STSBenchmark"] >= 0
    assert transforms[0].dataset is None  # Prepared data released after this task.


def test_retrieval_loads_corpus_queries_and_qrels_once(encoder_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    mteb = pytest.importorskip("mteb")
    from datasets import Dataset
    from mteb.abstasks import retrieval_dataset_loaders
    from mteb.abstasks.retrieval import RetrievalDatasetLoader

    loads = []

    def load(loader, **kwargs):
        loads.append(loader)
        return {
            "corpus": Dataset.from_dict({"id": ["d1", "d2", "d3", "d4"],
                                         "text": ["good day", "bad night", "sun", "rain"]}),
            "queries": Dataset.from_dict({"id": ["q1", "q2"], "text": ["good", "bad"]}),
            "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1}}, "top_ranked": None,
        }

    monkeypatch.setattr(retrieval_dataset_loaders, "get_dataset_config_names", lambda *a, **kw: ["default"])
    monkeypatch.setattr(RetrievalDatasetLoader, "load", load)
    model = AutoModel.from_pretrained(encoder_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    result = evaluate_model(model, tokenizer, [4, 16], tmp_path, eval_tasks=["ArguAna"], batch_size=2)
    assert len(loads) == 1
    assert all(np.isfinite(scores["ArguAna"]) for scores in result["scores"].values())


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
def test_default_eval_uses_only_selected_preset(method, pairs_csv):
    module = importlib.import_module(f"{method}.train")
    args = module.parse_args(["--model_key", "bert"])
    assert args.run_eval == 1 and tuple(args.eval_tasks) == DEFAULT_EVAL_TASKS
    args.data_path = str(pairs_csv)
    module.validate_args(args)
    override = module.parse_args(["--model_key", "bert", "--eval_tasks", "STSBenchmark", "SICK-R"])
    assert override.eval_tasks == ["STSBenchmark", "SICK-R"]
    args.eval_tasks = []
    with pytest.raises(ValueError, match="eval_tasks"):
        module.validate_args(args)


def test_selected_preset_resolves_exact_task_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    mteb = pytest.importorskip("mteb")
    assert DEFAULT_EVAL_TASKS and len(DEFAULT_EVAL_TASKS) == len(set(DEFAULT_EVAL_TASKS))
    for task_type, names in EVAL_TASK_GROUPS.items():
        for name in names:
            task = mteb.get_task(name, eval_splits=["test"], languages=["eng"])
            assert task.metadata.name == name
            assert task.metadata.type == task_type
            assert "test" in task.metadata.eval_splits and task.hf_subsets
