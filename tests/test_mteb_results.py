"""Reports retain every epoch and align task scores across embedding sizes."""

import csv
import json

from common.mteb_results import write_result_summaries


def save_epoch(directory, epoch, scores, *, metric="cosine_spearman"):
    result = {"backend": "mteb", "epoch": epoch, "tasks": ["STSBenchmark"],
              "metrics": {"STSBenchmark": metric}, "matryoshka_dims": list(scores),
              "scores": {f"dim_{dim}": {"STSBenchmark": score} for dim, score in scores.items()}}
    path = directory / (f"result_epoch_{epoch}.json" if epoch is not None else "result.json")
    path.write_text(json.dumps(result))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_epochs_sort_numerically_and_repeated_evaluation_replaces_scores(tmp_path):
    save_epoch(tmp_path, 10, {16: 0.82, 32: 0.85})
    save_epoch(tmp_path, 2, {16: 0.72, 32: 0.75})
    assert write_result_summaries(tmp_path) == 4
    rows = read_csv(tmp_path / "results.csv")
    assert [row["epoch"] for row in rows] == ["2", "10"]
    assert rows[0] == {"epoch": "2", "task": "STSBenchmark", "metric": "cosine_spearman",
                       "dim_16": "0.72", "dim_32": "0.75"}
    summary = (tmp_path / "summary_all_epochs.txt").read_text()
    assert "| Epoch | Metric | 16 | 32 |" in summary
    assert "| 2 | cosine_spearman | 0.7200 | 0.7500 |" in summary
    save_epoch(tmp_path, 2, {16: 0.76, 32: 0.79})
    assert write_result_summaries(tmp_path) == 4
    rows = read_csv(tmp_path / "results_long.csv")
    assert len(rows) == 4
    assert rows[0] == {"epoch": "2", "task": "STSBenchmark", "dim": "16",
                       "metric": "cosine_spearman", "score": "0.76"}


def test_tasks_and_missing_dimensions_remain_aligned(tmp_path):
    save_epoch(tmp_path, 1, {16: 0.5, 64: None})
    save_epoch(tmp_path, 2, {32: 0.6, 64: 0.7})
    path = tmp_path / "result_epoch_2.json"
    result = json.loads(path.read_text())
    result["tasks"].append("FiQA2018")
    result["metrics"]["FiQA2018"] = "ndcg_at_10"
    result["scores"]["dim_32"]["FiQA2018"] = 0.3
    result["scores"]["dim_64"]["FiQA2018"] = 0.4
    path.write_text(json.dumps(result))
    assert write_result_summaries(tmp_path) == 6
    rows = read_csv(tmp_path / "results.csv")
    assert [(row["epoch"], row["task"]) for row in rows] == [
        ("1", "STSBenchmark"), ("2", "STSBenchmark"), ("2", "FiQA2018")]
    assert rows[0]["dim_32"] == rows[0]["dim_64"] == ""
    assert rows[2]["metric"] == "ndcg_at_10" and rows[2]["dim_16"] == ""
    assert "| 1 | cosine_spearman | 0.5000 | N/A | N/A |" in (tmp_path / "summary_all_epochs.txt").read_text()


def test_standalone_and_existing_results_without_metric_names(tmp_path):
    save_epoch(tmp_path, None, {16: 0.65})
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    del result["metrics"]
    result_path.write_text(json.dumps(result))
    assert write_result_summaries(tmp_path) == 1
    row, = read_csv(tmp_path / "results_long.csv")
    assert row["epoch"] == "standalone" and row["metric"] == "main_score"
