#!/usr/bin/env bash
# Download the original Kaggle bundles next to FlagMRL/ and check their CSVs.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
FORCE=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1 ;;
        --check-only) CHECK_ONLY=1 ;;
        -h|--help)
            cat <<'HELP'
Usage: bash FlagMRL/download.sh [--force | --check-only]

  truongminhphuc/final-data -> <project>/final-data/
  winddao/multitask-data    -> <project>/multitask-data/
  truongminhphuc/sts-data   -> <project>/sts-data/

Skips datasets whose required CSVs already pass checks; --force downloads again.
--check-only checks local CSVs without Kaggle/network access. In this mode,
DATA_PATH and EVAL_WORK_DIR override paths, and RUN_EVAL=0 checks only train data.
PYTHON_BIN and KAGGLE_BIN can select the Python and Kaggle executables.
Authenticate Kaggle first (kaggle auth login, or existing API credentials).
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1. Use --help." >&2; exit 2 ;;
    esac
    shift
done

if [[ "$FORCE" == "1" && "$CHECK_ONLY" == "1" ]]; then
    echo "--force and --check-only cannot be combined." >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 127
fi

cd "$PROJECT_DIR"
if [[ "$CHECK_ONLY" == "1" ]]; then
    DATA_PATH="${DATA_PATH:-$PROJECT_DIR/final-data/final_data.csv}"
    EVAL_WORK_DIR="${EVAL_WORK_DIR:-$PROJECT_DIR/FlagMRL}"
    RUN_EVAL="${RUN_EVAL:-1}"
else
    DATA_PATH="$PROJECT_DIR/final-data/final_data.csv"
    EVAL_WORK_DIR="$PROJECT_DIR/FlagMRL"
    RUN_EVAL=1
fi
if [[ "$RUN_EVAL" != "0" && "$RUN_EVAL" != "1" ]]; then
    echo "RUN_EVAL must be 0 or 1." >&2
    exit 2
fi

check_files() {
    "$PYTHON_BIN" - "$PROJECT_DIR" "$DATA_PATH" "$EVAL_WORK_DIR" "$RUN_EVAL" "${1:-}" <<'PY'
import ast
import csv
from itertools import islice
from pathlib import Path
import sys

root = Path(sys.argv[1])
train_path = Path(sys.argv[2]).expanduser().resolve()
eval_dir = Path(sys.argv[3]).expanduser().resolve()
include_eval, folder = sys.argv[4] == "1", sys.argv[5]
files = [("train", train_path, ("text",), 2)]
if include_eval:
    names = {"TEST_CLS_TASKS", "TEST_STS_TASKS", "TEST_PAIR_TASKS"}
    tasks = {}
    tree = ast.parse((root / "FlagMRL/data_preparation.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    tasks[target.id] = ast.literal_eval(node.value)
    if tasks.keys() != names:
        raise ValueError("Could not read TEST_*_TASKS from data_preparation.py.")
    for pair in tasks["TEST_CLS_TASKS"]:
        files.extend(("classification", (eval_dir / p).resolve(), ("text", "label"), 1) for p in pair)
    files.extend(("sts", (eval_dir / p).resolve(), ("sentence1", "sentence2", "score"), 1)
                 for p in tasks["TEST_STS_TASKS"])
    files.extend(("pair", (eval_dir / p).resolve(), ("sentence1", "sentence2", "label"), 1)
                 for p in tasks["TEST_PAIR_TASKS"])
if folder:
    files = [spec for spec in files if spec[1].is_relative_to(root / folder)]
csv.field_size_limit(sys.maxsize)
failed = 0
for kind, path, columns, minimum in files:
    status = "OK"
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(columns) - set(reader.fieldnames or [])
            if missing:
                status = "INVALID: missing columns " + ", ".join(sorted(missing))
            elif len(list(islice(reader, minimum))) < minimum:
                status = f"INVALID: need at least {minimum} row(s)"
    except FileNotFoundError:
        status = "MISSING"
    except (OSError, UnicodeError, csv.Error) as error:
        status = f"INVALID: {error}"
    failed += int(status != "OK")
    if not folder:
        print(f"[{status}] {kind}: {path} | columns: {', '.join(columns)}")
if not folder:
    print(f"Required: {len(files)} CSV files; missing/invalid: {failed}.")
raise SystemExit(int(failed > 0 or not files))
PY
}

if [[ "$CHECK_ONLY" == "1" ]]; then
    check_files
    exit 0
fi

for DATASET in \
    "truongminhphuc/final-data" \
    "winddao/multitask-data" \
    "truongminhphuc/sts-data"; do
    FOLDER="${DATASET##*/}"
    if [[ "$FORCE" == "0" ]] && check_files "$FOLDER"; then
        echo "[SKIP] $DATASET: required CSVs already exist and pass checks."
        continue
    fi
    if ! command -v "$KAGGLE_BIN" >/dev/null 2>&1; then
        echo "Kaggle CLI not found: $KAGGLE_BIN. Install with: $PYTHON_BIN -m pip install kaggle" >&2
        exit 127
    fi
    DESTINATION="$PROJECT_DIR/$FOLDER"
    mkdir -p "$DESTINATION"
    DOWNLOAD_ARGS=(datasets download -d "$DATASET" --path "$DESTINATION" --unzip)
    if [[ "$FORCE" == "1" ]]; then
        DOWNLOAD_ARGS+=(--force)
    fi
    echo "[DOWNLOAD] $DATASET -> $DESTINATION"
    if "$KAGGLE_BIN" "${DOWNLOAD_ARGS[@]}"; then
        :
    else
        STATUS=$?
        echo "Download failed for $DATASET. Check Kaggle authentication/access and retry." >&2
        exit "$STATUS"
    fi
done

check_files
echo "Datasets ready. Run: bash FlagMRL/run_all.sh"
