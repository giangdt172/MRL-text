"""
Data preparation module - shared across all models.
Handles dataset classes, collators, and dataloader construction.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Training dataset
# ─────────────────────────────────────────────
class TextPairRaw(Dataset):
    def __init__(self, df: pd.DataFrame, task: str):
        self.task = task
        if task == "single_cls":
            self.samples = [(t, None, int(y)) for t, y in zip(df["text"].astype(str), df["label"].astype(int))]
        elif task == "pair_cls":
            self.samples = [(a, b) for a, b in zip(df["premise"].astype(str),
                                                    df["hypothesis"].astype(str))]
        else:  # pair_reg
            self.samples = [(a, b) for a, b in zip(df["sentence1"].astype(str),
                                                    df["sentence2"].astype(str))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# Dual-tokenizer collator (student + teacher)
# ─────────────────────────────────────────────
class DualTokenizerCollate:
    def __init__(self, tok_student, tok_teacher, task: str, max_len: int):
        self.ts = tok_student
        self.tt = tok_teacher
        self.task = task
        self.max_len = max_len

    def __call__(self, batch: List[Tuple[str, Optional[str], float]]):
        s1s, s2s = zip(*batch)

        if self.task == "single_cls":
            s_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                            padding=True, return_tensors="pt",
                            return_special_tokens_mask=True)
            t_enc = self.tt(list(s1s), max_length=self.max_len, truncation=True,
                            padding=True, return_tensors="pt",
                            return_special_tokens_mask=True)

            out = {
                "input_ids_stu": s_enc["input_ids"],
                "attention_mask_stu": s_enc["attention_mask"],
                "special_tokens_mask_stu": s_enc["special_tokens_mask"],
                "input_ids_tea": t_enc["input_ids"],
                "attention_mask_tea": t_enc["attention_mask"],
                "special_tokens_mask_tea": t_enc["special_tokens_mask"],
                "labels": torch.tensor(s1s, dtype=torch.long),
            }
            if "token_type_ids" in s_enc:
                out["token_type_ids_stu"] = s_enc["token_type_ids"]
            if "token_type_ids" in t_enc:
                out["token_type_ids_tea"] = t_enc["token_type_ids"]
            return out

        # ------- pair (bi-encoder) -------
        s1_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        s2_enc = self.ts(list(s2s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)

        t1_enc = self.tt(list(s1s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        t2_enc = self.tt(list(s2s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)

        out = {
            # student
            "input_ids1_stu": s1_enc["input_ids"],
            "attention_mask1_stu": s1_enc["attention_mask"],
            "special_tokens_mask1_stu": s1_enc["special_tokens_mask"],
            "input_ids2_stu": s2_enc["input_ids"],
            "attention_mask2_stu": s2_enc["attention_mask"],
            "special_tokens_mask2_stu": s2_enc["special_tokens_mask"],
            # teacher
            "input_ids1_tea": t1_enc["input_ids"],
            "attention_mask1_tea": t1_enc["attention_mask"],
            "special_tokens_mask1_tea": t1_enc["special_tokens_mask"],
            "input_ids2_tea": t2_enc["input_ids"],
            "attention_mask2_tea": t2_enc["attention_mask"],
            "special_tokens_mask2_tea": t2_enc["special_tokens_mask"],
        }
        if "token_type_ids" in s1_enc:
            out["token_type_ids1_stu"] = s1_enc["token_type_ids"]
        if "token_type_ids" in s2_enc:
            out["token_type_ids2_stu"] = s2_enc["token_type_ids"]
        if "token_type_ids" in t1_enc:
            out["token_type_ids1_tea"] = t1_enc["token_type_ids"]
        if "token_type_ids" in t2_enc:
            out["token_type_ids2_tea"] = t2_enc["token_type_ids"]

        return out


# ─────────────────────────────────────────────
# Evaluation datasets
# ─────────────────────────────────────────────
class STSDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instruction = ""
        return {
            "sentence1": instruction + self.dataset.iloc[idx]['sentence1'],
            "sentence2": instruction + self.dataset.iloc[idx]['sentence2'],
            "label": torch.tensor(self.dataset.iloc[idx]['score'], dtype=torch.float),
        }


class ClasssifyDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "text": self.dataset.iloc[idx]['text'],
            "label": torch.tensor(self.dataset.iloc[idx]['label'], dtype=torch.long),
        }


class PairDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instruction = ""
        return {
            "sentence1": instruction + self.dataset.iloc[idx]['sentence1'],
            "sentence2": instruction + self.dataset.iloc[idx]['sentence2'],
            "label": torch.tensor(self.dataset.iloc[idx]['label'], dtype=torch.float),
        }


# ─────────────────────────────────────────────
# Collate functions for evaluation
# ─────────────────────────────────────────────
def collate_fn(batch, tokenizer, max_len=128):
    s1_list = [item["sentence1"] for item in batch]
    s2_list = [item["sentence2"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(s1_list, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")
    enc2 = tokenizer(s2_list, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")

    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "input_ids2": enc2["input_ids"],
        "attention_mask2": enc2["attention_mask"],
        "labels": labels,
    }


def clf_collate_fn(batch, tokenizer, max_len=512):
    s1_list = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(s1_list, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")

    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "labels": labels,
    }


# ─────────────────────────────────────────────
# Build training dataloader
# ─────────────────────────────────────────────
def build_train_loader(
    data_path: str,
    model_name: str,
    task_type: str = "pair_cls",
    max_length: int = 256,
    batch_size: int = 16,
):
    df = pd.read_csv(data_path)

    # The training data has a 'text' column; map to premise/hypothesis for pair_cls
    df["premise"] = df["text"]
    df["hypothesis"] = df["text"]
    cols = ["premise", "hypothesis"]
    df_out = df[cols].copy()

    tok_student = AutoTokenizer.from_pretrained(model_name)
    tok_teacher = AutoTokenizer.from_pretrained(model_name)

    train_ds = TextPairRaw(df_out, task_type)
    collate = DualTokenizerCollate(tok_student, tok_teacher, task_type, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )

    print(f"Done Prepare Dataloader — {len(train_ds)} samples, {len(train_loader)} batches")
    return train_loader, tok_student, tok_teacher


# ─────────────────────────────────────────────
# Evaluation task path lists (Kaggle paths)
# ─────────────────────────────────────────────
EVAL_CLS_TASKS = [('../multitask-data/multi-data/banking_train.csv', 
                   '../multitask-data/multi-data/banking77_validation.csv'),
                  ('../multitask-data/multi-data/emotion_train.csv', 
                   '../multitask-data/multi-data/emotion_validation.csv'), 
                  ('../multitask-data/multi-data/tweet_train.csv', 
                   '../multitask-data/multi-data/tweet_validation.csv')]


EVAL_STS_TASKS = ['../multitask-data/multi-data/sick_validation.csv', 
                  '../multitask-data/multi-data/sts12_validation.csv', 
                  '../multitask-data/multi-data/stsb_validation.csv']

EVAL_PAIR_TASKS = ['../multitask-data/multi-data/mrpc_validation.csv', 
                   '../multitask-data/multi-data/scitail_validation.csv', 
                   '../multitask-data/multi-data/wic_validation.csv']

TEST_CLS_TASKS = [('../multitask-data/multi-data/banking_train.csv', 
                   '../multitask-data/multi-data/banking77_test.csv'),
                  ('../multitask-data/multi-data/emotion_train.csv', 
                   '../multitask-data/multi-data/emotion_test.csv'), 
                  ('../multitask-data/multi-data/tweet_train.csv', 
                   '../multitask-data/multi-data/tweet_test.csv')]


TEST_STS_TASKS = ['../multitask-data/multi-data/sick_test.csv', 
                  '../multitask-data/multi-data/sts12_test.csv', 
                  '../multitask-data/multi-data/stsb_test.csv',
                 '../sts-data/more_test_data/sick_r.csv', 
                  '../sts-data/more_test_data/sts13.csv', 
                  '../sts-data/more_test_data/sts14.csv',
                  '../sts-data/more_test_data/sts15.csv',
                  '../sts-data/more_test_data/sts16.csv']


TEST_PAIR_TASKS = ['../multitask-data/multi-data/mrpc_test.csv', 
                   '../multitask-data/multi-data/scitail_test.csv', 
                   '../multitask-data/multi-data/wic_test.csv']

