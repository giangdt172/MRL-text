"""Positive Quora pairs shared by MRL, MIPIC, FlagMRL, and ESE.

The Hub's pair/train split contains 149,263 pairs (approximately 150k).
Keep both sentences: positives come from annotations, not duplicated text.
Other pairs' positive sentences are in-batch negatives. Repeated questions can
therefore create false negatives; this loader does not change loss masking.
Evaluation uses selected task datasets through common.mteb_eval and MTEB.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


DATASET_NAME = "sentence-transformers/quora-duplicates"
DATASET_CONFIG = "pair"
DATASET_SPLIT = "train"
MAX_TRAIN_SAMPLES = 150_000


def add_training_data_args(parser):
    parser.add_argument("--data_path", default=None,
                        help="Optional local positive-pair CSV with anchor,positive columns; "
                             "default loads sentence-transformers/quora-duplicates (pair/train)")
    parser.add_argument("--max_train_samples", type=int, default=MAX_TRAIN_SAMPLES,
                        help="Cap on training pairs; use all available rows when fewer (no oversampling)")
    parser.add_argument("--data_cache_dir", default=None, help="Hugging Face dataset cache directory")
    parser.add_argument("--dataset_revision", default=None, help="Optional Hub dataset commit/tag")


def validate_training_data_args(args):
    if args.max_train_samples < 2:
        raise ValueError("max_train_samples must be at least two for in-batch negatives.")
    if args.data_path:
        path = Path(args.data_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training pair CSV not found: {path}")
        if not {"anchor", "positive"}.issubset(pd.read_csv(path, nrows=0).columns):
            raise ValueError("Training CSV must contain 'anchor' and 'positive' columns; "
                             "single-text and labeled pair-class CSVs are not positive-pair data.")
        args.data_path = str(path)
    else:
        args.data_path = None


class PositivePairDataset(Dataset):
    def __init__(self, frame, *, max_train_samples=MAX_TRAIN_SAMPLES, seed=42, metadata=None):
        if max_train_samples < 2:
            raise ValueError("max_train_samples must be at least two.")
        if not {"anchor", "positive"}.issubset(frame.columns):
            raise ValueError("Training data must have 'anchor' and 'positive' columns.")
        if len(frame) < 2:
            raise ValueError("Contrastive training needs at least two positive pairs.")
        for column in ("anchor", "positive"):
            valid = frame[column].map(lambda text: isinstance(text, str) and bool(text.strip()))
            if not valid.all():
                raise ValueError(f"Training '{column}' must contain nonempty strings without null values.")
        source_rows = len(frame)
        if source_rows > max_train_samples:
            frame = frame.sample(n=max_train_samples, random_state=seed)
        self.pairs = list(frame[["anchor", "positive"]].itertuples(index=False, name=None))
        digest = hashlib.sha256()
        for pair in self.pairs:
            digest.update((json.dumps(pair, ensure_ascii=False) + "\n").encode("utf-8"))
        self.metadata = {**(metadata or {}), "source_rows": source_rows, "num_pairs": len(self.pairs),
                         "max_train_samples": max_train_samples, "selection_seed": seed,
                         "pairs_sha256": digest.hexdigest(), "positive_source": "annotated_duplicate_pair",
                         "negative_source": "other_positive_columns_in_batch",
                         "deduplicated": False, "oversampled": False}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


class PositivePairCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, pairs):
        anchors, positives = zip(*pairs)
        # Encode each sentence independently; preserve token_type_ids if present.
        return {side: self.tokenizer(list(texts), padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                for side, texts in (("anchor", anchors), ("positive", positives))}


def batch_inputs(batch, side, device):
    key = {1: "anchor", 2: "positive"}[side]
    return {name: value.to(device, non_blocking=device.type == "cuda")
            for name, value in batch[key].items()}


def load_training_pairs(data_path=None, *, seed=42, max_train_samples=MAX_TRAIN_SAMPLES,
                        data_cache_dir=None, dataset_revision=None):
    if data_path:
        frame = pd.read_csv(data_path, keep_default_na=False,
                            dtype={"anchor": str, "positive": str})
        metadata = {"source": "local_csv", "path": str(Path(data_path).expanduser().resolve())}
    else:
        # Optional at import time, so local CSV training and evaluation also work offline.
        from datasets import load_dataset

        raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT,
                           cache_dir=data_cache_dir, revision=dataset_revision)
        frame = raw.to_pandas()
        metadata = {"source": DATASET_NAME, "config": DATASET_CONFIG, "split": DATASET_SPLIT,
                    "revision": dataset_revision, "hf_fingerprint": raw._fingerprint}
    return PositivePairDataset(frame, max_train_samples=max_train_samples, seed=seed, metadata=metadata)


def build_train_loader(data_path, model_name, max_length=256, batch_size=16, num_workers=2, *,
                       seed=42, max_train_samples=MAX_TRAIN_SAMPLES, data_cache_dir=None,
                       dataset_revision=None):
    if batch_size < 2 or max_length < 1 or num_workers < 0:
        raise ValueError("batch_size must be at least two, max_length positive, num_workers nonnegative.")
    dataset = load_training_pairs(data_path, seed=seed, max_train_samples=max_train_samples,
                                  data_cache_dir=data_cache_dir, dataset_revision=dataset_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token for batching.")
        tokenizer.pad_token = tokenizer.eos_token
    # A dedicated generator makes batch order independent of model initialization.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=PositivePairCollator(tokenizer, max_length),
                        num_workers=num_workers, persistent_workers=num_workers > 0,
                        pin_memory=torch.cuda.is_available())
    return loader, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_training_data_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", help="Optional destination for an anchor,positive CSV export")
    args = parser.parse_args()
    validate_training_data_args(args)
    pairs = load_training_pairs(args.data_path, seed=args.seed, max_train_samples=args.max_train_samples,
                                data_cache_dir=args.data_cache_dir, dataset_revision=args.dataset_revision)
    if args.output_csv:
        destination = Path(args.output_csv).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pairs.pairs, columns=["anchor", "positive"]).to_csv(destination, index=False)
        print(f"Exported positive pairs to {destination}")
    print(json.dumps(pairs.metadata, indent=2))
