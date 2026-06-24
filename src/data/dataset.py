"""PyTorch Dataset for AG News: reads the prepared Parquet and tokenizes for DistilBERT.

Why this module exists:
    The model does not read text, it reads token IDs. This Dataset is the bridge
    between the on-disk Parquet (text + label) and the tensors the training loop
    consumes. Texts are tokenized once up front (the subset is small enough to fit
    in memory), so each __getitem__ is a cheap index lookup.

    Tokenization uses a fixed max_length of 128: the EDA showed >99% of snippets
    are far shorter, so 128 covers them while keeping CPU training fast.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import MAX_LENGTH


class AGNewsDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ) -> None:
        df = pd.read_parquet(parquet_path)

        encoded = tokenizer(
            df["text"].tolist(),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Keys match what HuggingFace classification models expect, so the trainer
        # can call model(**batch) and get the loss back directly.
        self.input_ids = torch.tensor(encoded["input_ids"])
        self.attention_mask = torch.tensor(encoded["attention_mask"])
        self.labels = torch.tensor(df["label"].tolist())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
