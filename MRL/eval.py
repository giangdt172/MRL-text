"""Evaluate MRL checkpoints on selected MTEB tasks at Matryoshka dimensions."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.mteb_eval import evaluate as _evaluate, evaluate_model, main


def evaluate(checkpoint_dir=None, **kwargs):
    return _evaluate(checkpoint_dir, method="mrl", **kwargs)


if __name__ == "__main__":
    main("mrl")
