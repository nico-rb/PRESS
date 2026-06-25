"""Thin Azure Blob Storage client: upload/download a single artifact.

This is the only training/serving-side module that holds cloud credentials. It reads
an Azure Storage connection string from the AZURE_STORAGE_CONNECTION_STRING
environment variable (kept in a non-versioned .env), so the rest of the code stays
cloud-agnostic and never imports the Azure SDK directly.

Why "blob": "blob" = Binary Large OBject, the cloud-storage term for an opaque byte
payload stored under a name (here, model.pt). In Azure, blobs live inside *containers*
within a *storage account*; one blob is one file. The name mirrors AWS S3 "objects"
and keeps this module a direct analogue of TR-ACE's aws_client.py.

Loading the .env: this module only reads os.environ, so load the .env before calling
it, e.g. `uv run --env-file .env python -m scripts.blob_roundtrip`.
"""

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient

_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"


def _service_client() -> BlobServiceClient:
    conn = os.environ.get(_CONNECTION_STRING_ENV)
    if not conn:
        raise RuntimeError(
            f"{_CONNECTION_STRING_ENV} is not set. Put it in a non-versioned .env and "
            "load it (e.g. `uv run --env-file .env ...`) before using azure_blob."
        )
    return BlobServiceClient.from_connection_string(conn)


def upload_blob(local_path: str | Path, container: str, blob_name: str) -> None:
    """Upload a local file to <container>/<blob_name>, creating the container if absent."""
    client = _service_client().get_container_client(container)
    if not client.exists():
        client.create_container()
    with open(local_path, "rb") as f:
        client.upload_blob(name=blob_name, data=f, overwrite=True)


def download_blob(container: str, blob_name: str, local_path: str | Path) -> None:
    """Download <container>/<blob_name> to a local file, creating parent dirs."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob = _service_client().get_container_client(container).get_blob_client(blob_name)
    with open(local_path, "wb") as f:
        f.write(blob.download_blob().readall())
