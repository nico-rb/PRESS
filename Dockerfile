# Serving image for the AG News classifier. Build from the repo root:
#   docker build -t press-serving .

FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch + the serving stack. On Linux the default torch wheel bundles
# CUDA (gigabytes); we serve on CPU, so install the CPU build from PyTorch's index.
# pyyaml is needed because main.py -> src.config imports it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers fastapi uvicorn pyyaml

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
