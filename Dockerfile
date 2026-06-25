# Serving image for the AG News classifier. Build from the repo root:
#   docker build -t press-serving .

FROM python:3.12-slim

WORKDIR /app

# PyTorch + the serving stack. We install the default torch wheel, which on Linux
# bundles CUDA, so the container uses a GPU when one is exposed to it
# (`docker run --gpus all` on a host with NVIDIA drivers). main.py auto-detects the
# device, so with no GPU the same image falls back to CPU. The wheel is several GB
# because of the bundled CUDA libraries. pyyaml is needed because main.py ->
# src.config imports it.
RUN pip install --no-cache-dir torch transformers fastapi uvicorn pyyaml

# Bake the tokenizer + base DistilBERT into the image's HuggingFace cache, so the
# container needs no network at runtime (our trained weights override these later).
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)"

# Application code + the trained weights.
COPY src/ src/
COPY main.py main.py
COPY outputs/models/1_epoch/model.pt outputs/models/1_epoch/model.pt

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container (not just localhost).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
