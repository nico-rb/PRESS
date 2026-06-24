"""Experiment configuration: a typed TrainingConfig loaded from a YAML file.

Why config-driven:
    All knobs for one experiment live in a single YAML under configs/experiments/.
    train.py reads a config into this dataclass and runs; trying a different setup
    (learning rate, epochs, ...) means writing another YAML, not touching code.
    That keeps experiments reproducible and easy to track.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Shared domain constants. Per-experiment YAMLs override these via TrainingConfig;
# serving and the default factory/dataset arguments use them directly.
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LENGTH = 128
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class TrainingConfig:
    # data
    train_path: str
    val_path: str
    # model
    model_name: str
    num_labels: int
    max_length: int
    # training
    batch_size: int
    learning_rate: float
    epochs: int
    seed: int
    # output
    output_dir: str


def load_config(path: str | Path) -> TrainingConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**data)
