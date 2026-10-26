"""
Evaluation module - shared across all models.
Handles STS, classification, and pair-classification evaluation.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, average_precision_score)
from sklearn.linear_model import LogisticRegression

from .data_preparation import (
    STSDataset, ClasssifyDataset, PairDataset,
    collate_fn, clf_collate_fn,
)


# ─────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────
def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output.last_hidden_state  # [B, L, D]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)  # [B, D]
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # [B, D]
    mean_embeddings = sum_embeddings / sum_mask  # [B, D]
    return mean_embeddings


# ─────────────────────────────────────────────
# STS evaluation
# ─────────────────────────────────────────────
def eval_sts(model, eval_loader, matryoshka_dims):
    results = {}
    device = model.device

    for dim in matryoshka_dims:
        preds, labels = [], []

        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Eval dim={dim}"):
                    input_ids1 = batch["input_ids1"].to(device)
                    attn1 = batch["attention_mask1"].to(device)
                    input_ids2 = batch["input_ids2"].to(device)
                    attn2 = batch["attention_mask2"].to(device)
                    label = batch["labels"]

                    out1 = model(input_ids=input_ids1, attention_mask=attn1)
                    out2 = model(input_ids=input_ids2, attention_mask=attn2)

                    emb1 = mean_pooling(out1, attn1)[:, :dim]
                    emb2 = mean_pooling(out2, attn2)[:, :dim]

                    sim = F.cosine_similarity(emb1, emb2)
                    score = (sim + 1) * 2.5  # scale [-1,1] -> [0,5]

                    preds.extend(score.cpu().numpy())
                    labels.extend(label.numpy())

        spearman_corr, _ = spearmanr(preds, labels)
        results[f"dim_{dim}"] = spearman_corr
        print(f"  Dim {dim}: Spearman = {spearman_corr:.4f}")

    return results


def eval_sts_task(model, tokenizer, path_list, matryoshka_dims):
    task_results = {}
    model.eval()
    print(' ✅ eval_sts_task')
    for path in path_list:
        print(f"\n{path}")
        eval_dataset = STSDataset(path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, tokenizer),
        )
        results = eval_sts(model, eval_loader, matryoshka_dims)
        task_results[Path(path).stem] = results
    model.train()
    return task_results


# ─────────────────────────────────────────────
# Classification evaluation
# ─────────────────────────────────────────────
def eval_cls(model, eval_loader, dim):
    preds, labels = [], []
    device = model.device

    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Extract dim={dim}"):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                label = batch["labels"]

                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                emb1 = mean_pooling(out1, attn1)[:, :dim]

                preds.extend(emb1.cpu().numpy())
                labels.extend(label.numpy())

    return preds, labels


def eval_classification_task(model, tokenizer, path_list, matryoshka_dims):
    task_results = {}
    model.eval()
    print(' ✅ eval classifier')

    for train_path, dev_path in path_list:
        dataset_results = {}
        print(f"\n{dev_path}")

        for dim in matryoshka_dims:
            eval_dataset = ClasssifyDataset(dev_path)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=lambda x: clf_collate_fn(x, tokenizer),
            )

            train_dataset = ClasssifyDataset(train_path)
            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=lambda x: clf_collate_fn(x, tokenizer),
            )

            X_train, y_train = eval_cls(model, train_loader, dim=dim)
            X_test, y_test = eval_cls(model, eval_loader, dim=dim)

            clf = LogisticRegression(random_state=42, n_jobs=1, max_iter=200, verbose=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            dataset_results[f"dim_{dim}"] = {"accuracy": accuracy, "f1": f1}
            print(f"  Dim {dim}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

        task_results[Path(dev_path).stem] = dataset_results
    model.train()
    return task_results


# ─────────────────────────────────────────────
# Pair classification evaluation
# ─────────────────────────────────────────────
def get_metric_pair_classification(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    best_acc, best_thr = 0, 0
    for thr in np.linspace(0, 1, 200):
        preds = (scores >= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    preds = (scores >= best_thr).astype(int)
    return {
        "best_threshold": best_thr,
        "accuracy": best_acc,
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "average_precision": average_precision_score(labels, scores),
    }


def eval_pair(model, eval_loader, matryoshka_dims):
    results = {}
    device = model.device

    for dim in matryoshka_dims:
        preds, labels = [], []

        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Eval dim={dim}"):
                    input_ids1 = batch["input_ids1"].to(device)
                    attn1 = batch["attention_mask1"].to(device)
                    input_ids2 = batch["input_ids2"].to(device)
                    attn2 = batch["attention_mask2"].to(device)
                    label = batch["labels"]

                    out1 = model(input_ids=input_ids1, attention_mask=attn1)
                    out2 = model(input_ids=input_ids2, attention_mask=attn2)

                    emb1 = mean_pooling(out1, attn1)[:, :dim]
                    emb2 = mean_pooling(out2, attn2)[:, :dim]

                    sim = F.cosine_similarity(emb1, emb2)
                    score = (sim + 1) / 2

                    preds.extend(score.cpu().numpy())
                    labels.extend(label.numpy())

        metric = get_metric_pair_classification(preds, labels)
        results[f"dim_{dim}"] = metric
        print(f"  Dim {dim}: {metric}")

    return results


def eval_pair_task(model, tokenizer, path_list, matryoshka_dims):
    task_results = {}
    model.eval()
    print(' ✅ eval_pair_task')
    for path in path_list:
        print(f"\n{path}")
        eval_dataset = PairDataset(path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, tokenizer),
        )
        results = eval_pair(model, eval_loader, matryoshka_dims)
        task_results[Path(path).stem] = results
    model.train()
    return task_results
