"""Entrypoint: fine-tune DistilBERT on AG News from a YAML config.

Usage (from the repo root):
    uv run python train.py [configs/experiments/baseline.yaml]
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import AGNewsDataset
from src.models.factory import build_model
from src.trainers.classifier_trainer import ClassifierTrainer

DEFAULT_CONFIG = "configs/experiments/baseline.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Use the GPU if one is available, otherwise the CPU. Same code runs on both."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)
    set_seed(config.seed)

    device = get_device()
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    pin = device.type == "cuda"
    train_ds = AGNewsDataset(config.train_path, tokenizer, config.max_length)
    val_ds = AGNewsDataset(config.val_path, tokenizer, config.max_length)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, pin_memory=pin)

    model = build_model(config.model_name, config.num_labels)
    trainer = ClassifierTrainer(model, train_loader, val_loader, config.learning_rate, device)
    history = trainer.train(config.epochs)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved model   -> {out_dir / 'model.pt'}")
    print(f"Saved metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
