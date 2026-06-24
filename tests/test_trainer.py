from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.trainers.classifier_trainer import ClassifierTrainer

NUM_LABELS = 4
VOCAB = 20
SEQ_LEN = 8


class TinyModel(nn.Module):
    """Minimal stand-in for a HF classification model: returns .loss and .logits."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 8)
        self.classifier = nn.Linear(8, NUM_LABELS)

    def forward(self, input_ids, attention_mask=None, labels=None):
        pooled = self.embedding(input_ids).mean(dim=1)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


class LearnableDataset(Dataset):
    """Each example's tokens encode its label, so the model can actually learn it."""

    def __init__(self, n=32):
        self.labels = torch.arange(n) % NUM_LABELS
        self.input_ids = self.labels[:, None].repeat(1, SEQ_LEN)
        self.attention_mask = torch.ones(n, SEQ_LEN, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


@pytest.fixture
def trainer():
    torch.manual_seed(0)
    ds = LearnableDataset()
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds, batch_size=8)
    model = TinyModel()
    return ClassifierTrainer(
        model, train_loader, val_loader, learning_rate=1e-2, device=torch.device("cpu")
    )


def test_train_returns_one_record_per_epoch(trainer):
    history = trainer.train(epochs=3)
    assert len(history) == 3
    assert set(history[0]) == {"epoch", "train_loss", "accuracy", "macro_f1"}


def test_evaluate_metrics_within_unit_range(trainer):
    metrics = trainer.evaluate()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_training_reduces_loss(trainer):
    history = trainer.train(epochs=5)
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_training_updates_parameters(trainer):
    before = trainer.model.classifier.weight.clone()
    trainer.train(epochs=1)
    assert not torch.equal(before, trainer.model.classifier.weight)
