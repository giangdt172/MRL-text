"""Evaluation datasets and collators; training preparation lives in common/."""

import pandas as pd
import torch
from torch.utils.data import Dataset


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
# Evaluation paths relative to ESE/
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

