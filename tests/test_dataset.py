import pandas as pd
import pytest
from transformers import AutoTokenizer

from src.data.dataset import AGNewsDataset

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 16


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def parquet_path(tmp_path):
    df = pd.DataFrame(
        {
            "text": [
                "stocks rally on wall street",
                "the team won the championship",
                "new telescope discovers a planet",
                "diplomats meet to discuss peace",
                "company reports record profits",
                "athlete breaks world record",
            ],
            "label": [2, 1, 3, 0, 2, 1],
        }
    )
    path = tmp_path / "mini.parquet"
    df.to_parquet(path)
    return path


def test_dataset_length_matches_rows(parquet_path, tokenizer):
    ds = AGNewsDataset(parquet_path, tokenizer, max_length=MAX_LENGTH)
    assert len(ds) == 6


def test_dataset_item_has_model_keys(parquet_path, tokenizer):
    ds = AGNewsDataset(parquet_path, tokenizer, max_length=MAX_LENGTH)
    assert set(ds[0].keys()) == {"input_ids", "attention_mask", "labels"}


@pytest.mark.parametrize("key", ["input_ids", "attention_mask"])
def test_dataset_sequence_tensors_have_max_length_shape(parquet_path, tokenizer, key):
    ds = AGNewsDataset(parquet_path, tokenizer, max_length=MAX_LENGTH)
    assert ds[0][key].shape == (MAX_LENGTH,)


def test_dataset_label_matches_source(parquet_path, tokenizer):
    ds = AGNewsDataset(parquet_path, tokenizer, max_length=MAX_LENGTH)
    assert ds[2]["labels"].item() == 3
