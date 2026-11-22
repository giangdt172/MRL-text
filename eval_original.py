"""Evaluate an original encoder on selected MTEB tasks at Matryoshka dimensions."""

from common.mteb_eval import evaluate as _evaluate, main


def evaluate(**kwargs):
    return _evaluate(method="original", **kwargs)


if __name__ == "__main__":
    main("original")
