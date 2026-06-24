"""Model factory: build the DistilBERT classifier from a model name.

Why a factory:
    Centralizes model creation in one place. train.py just calls build_model(...)
    and never touches HuggingFace details, so swapping the model later is a one-line
    change here. We don't need a custom nn.Module subclass: for a standard model
    AutoModelForSequenceClassification already gives us DistilBERT plus a 4-class
    classification head.
"""

from transformers import AutoModelForSequenceClassification, PreTrainedModel

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4


def build_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
