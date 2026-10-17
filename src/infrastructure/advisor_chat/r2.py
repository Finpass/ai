import logging
from collections.abc import Iterator
from dataclasses import dataclass

import boto3
from botocore.config import Config

from src.core.config import settings

logger = logging.getLogger("rag.r2")


@dataclass(frozen=True)
class RemoteFile:
    key: str
    etag: str
    size: int


def client():
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def list_pdfs(prefix: str = "") -> Iterator[RemoteFile]:
    pages = (
        client().get_paginator("list_objects_v2").paginate(Bucket=settings.r2_bucket, Prefix=prefix)
    )
    for page in pages:
        for o in page.get("Contents", []):
            if o["Key"].lower().endswith(".pdf"):
                yield RemoteFile(o["Key"], o["ETag"].strip('"'), o["Size"])


def download(key: str, dest: str) -> None:
    client().download_file(settings.r2_bucket, key, dest)
