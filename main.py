"""FastAPI serving app: POST /predict classifies a news snippet with the trained model.

The model (DistilBERT + classification head) and tokenizer are loaded once at startup
and reused for every request. The device is auto-detected (GPU if available, else CPU).
"""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.config import CLASS_NAMES, MAX_LENGTH, MODEL_NAME, NUM_LABELS
from src.models.factory import build_model

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models/1_epoch/model.pt")

# Populated at startup, reused across requests.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MODEL_NAME, NUM_LABELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    state["model"] = model
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    state["device"] = device
    yield
    state.clear()


app = FastAPI(title="PRESS", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    class_name: str
    confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    encoding = state["tokenizer"](
        req.text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    ).to(state["device"])

    with torch.no_grad():
        logits = state["model"](**encoding).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = int(probs.argmax())

    return PredictResponse(
        label=label,
        class_name=CLASS_NAMES[label],
        confidence=round(probs[label].item(), 4),
    )
