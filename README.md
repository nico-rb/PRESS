# PRESS

**Pipeline for Reproducible End-to-end Serving System** — an end-to-end MLOps
pipeline on Azure.

A model that trains in a notebook is not a system. The work that makes ML
real — reproducible training, tracked experiments, versioned artifacts,
packaging, deployment, and automation. Without it, runs aren't reproducible 
and nothing actually ships.

## Problem

News is published faster than anyone can read it. To organize, route, or filter
that stream, articles first have to be sorted by topic — is a story about the
World, Sports, Business, or Sci/Tech? Doing that by hand doesn't scale, which
makes topic labeling a natural job for a text classifier.

## Solution

PRESS fine-tunes a [DistilBERT](https://huggingface.co/distilbert-base-uncased)
classifier on the [AG News](https://huggingface.co/datasets/ag_news) dataset to tag a piece of
news text with one of those four topics, and serves it as a REST API. The model is
kept deliberately small — the real subject of the project is the engineering
*around* it: reproducible training, experiment tracking, versioned artifacts,
containerized serving, and deployment on Azure.


## Tech stack

| Concern | Tool |
| --- | --- |
| Modeling | PyTorch, HuggingFace `transformers` + `datasets` |
| Training | Config-driven (dataclass + YAML), GPU if available else CPU |
| Experiment tracking | MLflow |
| Serving | FastAPI + Docker |
| Cloud storage | Azure Blob Storage |
| Cloud training | Azure Machine Learning |
| Deployment | Azure Container Registry + Azure Container Apps |
| CI/CD | GitHub Actions *(planned)* |
| Infrastructure | Bicep / Terraform *(planned)* |
