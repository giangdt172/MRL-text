"""Build readable epoch/task/dimension reports from saved MTEB results.

Rebuild existing reports without running evaluation:
    python -m common.mteb_results checkpoints/mipic/bert-base-uncased
"""

import argparse
import csv
import json
import math
from pathlib import Path


def format_score(score):
    return "N/A" if score is None or not math.isfinite(score) else f"{score:.4f}"


def write_csv(path, columns, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_result_summaries(output_dir):
    """Rebuild from epoch snapshots, so re-evaluating an epoch replaces its rows."""
    directory = Path(output_dir).expanduser().resolve()
    history = [json.loads(path.read_text()) for path in directory.glob("result_epoch_*.json")]
    if not history and (directory / "result.json").is_file():
        history = [json.loads((directory / "result.json").read_text())]
    history = [result for result in history if result.get("backend") == "mteb"]
    if not history:
        return 0
    history.sort(key=lambda result: (result["epoch"] is None, result["epoch"] or 0))
    tasks = list(dict.fromkeys(task for result in history for task in result["tasks"]))
    dims = sorted({dim for result in history for dim in result["matryoshka_dims"]})
    wide_rows, long_rows = [], []
    tables = ["MTEB results across epochs", "",
              "Each task uses its own main metric on the original MTEB score scale.",
              "N/A means missing or non-finite. Standalone means evaluation without an epoch number.", ""]
    for task in tasks:
        tables.extend([task, "", "| Epoch | Metric | " + " | ".join(str(dim) for dim in dims) + " |",
                       "| ---: | --- | " + " | ".join("---:" for _ in dims) + " |"])
        for result in history:
            if task not in result["tasks"]:
                continue
            epoch = result["epoch"] if result["epoch"] is not None else "standalone"
            metric = result.get("metrics", {}).get(task, "main_score")
            row = {"epoch": epoch, "task": task, "metric": metric}
            display = []
            for dim in dims:
                scores = result["scores"].get(f"dim_{dim}", {})
                score = scores.get(task)
                value = score if score is not None and math.isfinite(score) else ""
                row[f"dim_{dim}"] = value
                display.append(format_score(score))
                if task in scores:
                    long_rows.append({"epoch": epoch, "task": task, "dim": dim, "metric": metric, "score": value})
            wide_rows.append(row)
            tables.append(f"| {epoch} | {metric} | " + " | ".join(display) + " |")
        tables.append("")
    # CSV is ordered by epoch first, then the user's task order, then dimension.
    task_order = {task: index for index, task in enumerate(tasks)}
    order = lambda row: (row["epoch"] == "standalone", row["epoch"] if row["epoch"] != "standalone" else 0,
                         task_order[row["task"]], row.get("dim", 0))
    write_csv(directory / "results.csv", ["epoch", "task", "metric"] + [f"dim_{dim}" for dim in dims],
              sorted(wide_rows, key=order))
    write_csv(directory / "results_long.csv", ["epoch", "task", "dim", "metric", "score"],
              sorted(long_rows, key=order))
    summary_path = directory / "summary_all_epochs.txt"
    temporary = summary_path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(tables), encoding="utf-8")
    temporary.replace(summary_path)
    return len(long_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Checkpoint/results directory containing result_epoch_*.json")
    args = parser.parse_args()
    count = write_result_summaries(args.output_dir)
    print(f"Summarized {count} scores in {Path(args.output_dir).resolve()}")
