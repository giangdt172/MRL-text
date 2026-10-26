"""Classification, pair classification and STS using the checkpoint's pooling."""

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, average_precision_score,
)

from .data_preparation import STSDataset, ClasssifyDataset, PairDataset, collate_fn, clf_collate_fn
from .loss import pool_hidden


def mean_pooling(model_output, attention_mask):
    return pool_hidden(model_output.last_hidden_state, attention_mask, "mean")


def extract_embeddings(model, loader, pair=False, pooling="cls"):
    device = model.device
    first, second, labels = [], [], []
    was_training = model.training
    model.eval()
    dtype = torch.float16
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            for batch in tqdm(loader, desc="Extract embeddings"):
                for side, target in ((1, first), (2, second)) if pair else ((1, first),):
                    inputs = {key: batch[f"{key}{side}"].to(device)
                              for key in ("input_ids", "attention_mask", "token_type_ids")
                              if f"{key}{side}" in batch}
                    output = model(**inputs, return_dict=True, output_hidden_states=False)
                    target.append(pool_hidden(output.last_hidden_state, inputs["attention_mask"], pooling).cpu())
                labels.append(batch["labels"].cpu())
    finally:
        model.train(was_training)
    if not first:
        raise ValueError("Evaluation dataset is empty.")
    return torch.cat(first), torch.cat(second) if pair else None, torch.cat(labels).numpy()


def eval_cls(model, eval_loader, dim, pooling="cls"):
    embeddings, _, labels = extract_embeddings(model, eval_loader, pooling=pooling)
    return embeddings[:, :dim].numpy(), labels


def eval_sts(model, eval_loader, matryoshka_dims, pooling="cls"):
    first, second, labels = extract_embeddings(model, eval_loader, pair=True, pooling=pooling)
    results = {}
    for dim in matryoshka_dims:
        scores = ((F.cosine_similarity(first[:, :dim], second[:, :dim]) + 1) * 2.5).numpy()
        corr = float(spearmanr(scores, labels).statistic)
        results[f"dim_{dim}"] = corr
        print(f"  Dim {dim}: Spearman = {corr:.4f}")
    return results


def get_metric_pair_classification(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    best_acc, best_thr = -1, 0
    # Keep the same threshold search protocol as MRL/FlagMRL.
    for threshold in np.linspace(0, 1, 200):
        accuracy = accuracy_score(labels, scores >= threshold)
        if accuracy > best_acc:
            best_acc, best_thr = accuracy, threshold
    predictions = (scores >= best_thr).astype(int)
    return {
        "best_threshold": best_thr, "accuracy": best_acc,
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "average_precision": average_precision_score(labels, scores),
    }


def eval_pair(model, eval_loader, matryoshka_dims, pooling="cls"):
    first, second, labels = extract_embeddings(model, eval_loader, pair=True, pooling=pooling)
    results = {}
    for dim in matryoshka_dims:
        scores = ((F.cosine_similarity(first[:, :dim], second[:, :dim]) + 1) / 2).numpy()
        metric = get_metric_pair_classification(scores, labels)
        results[f"dim_{dim}"] = metric
        print(f"  Dim {dim}: {metric}")
    return results


def eval_classification_task(model, tokenizer, path_list, matryoshka_dims, pooling="cls", batch_size=64, max_length=512):
    results = {}
    for train_path, test_path in path_list:
        print(f"\n{test_path}")
        def loader(path):
            return DataLoader(ClasssifyDataset(path), batch_size=batch_size, shuffle=False,
                              collate_fn=partial(clf_collate_fn, tokenizer=tokenizer, max_len=max_length))
        train_embeddings, _, y_train = extract_embeddings(model, loader(train_path), pooling=pooling)
        test_embeddings, _, y_test = extract_embeddings(model, loader(test_path), pooling=pooling)
        scores = {}
        for dim in matryoshka_dims:
            classifier = LogisticRegression(random_state=42, n_jobs=1, max_iter=200)
            classifier.fit(train_embeddings[:, :dim].numpy(), y_train)
            predictions = classifier.predict(test_embeddings[:, :dim].numpy())
            scores[f"dim_{dim}"] = {
                "accuracy": accuracy_score(y_test, predictions),
                "f1": f1_score(y_test, predictions, average="macro", zero_division=0),
            }
            print(f"  Dim {dim}: {scores[f'dim_{dim}']}")
        results[Path(test_path).stem] = scores
    return results


def eval_pair_task(model, tokenizer, path_list, matryoshka_dims, pooling="cls", batch_size=64, max_length=128):
    results = {}
    for path in path_list:
        print(f"\n{path}")
        loader = DataLoader(PairDataset(path), batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, max_len=max_length))
        results[Path(path).stem] = eval_pair(model, loader, matryoshka_dims, pooling)
    return results


def eval_sts_task(model, tokenizer, path_list, matryoshka_dims, pooling="cls", batch_size=64, max_length=128):
    results = {}
    for path in path_list:
        print(f"\n{path}")
        loader = DataLoader(STSDataset(path), batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer, max_len=max_length))
        results[Path(path).stem] = eval_sts(model, loader, matryoshka_dims, pooling)
    return results
