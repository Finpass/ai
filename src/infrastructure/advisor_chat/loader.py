import json
import os
from importlib.metadata import version
from pathlib import Path

import boto3

from src.core.config import settings


def ensure_corpus():
    """
    Fungsi ini dipanggil satu kali saat aplikasi FastAPI start (di lifespan).
    Mengunduh indeks ChromaDB dari R2 bila belum ada, dan memvalidasi versinya.
    """
    prefix = os.getenv("CORPUS_PREFIX", "artifacts/chroma_corpus_v1/")
    bucket = os.getenv("R2_BUCKET")
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    expected_embed_model = os.getenv("EMBED_MODEL", "baai/bge-m3")

    chroma_path = Path(settings.chroma_path)
    manifest_path = chroma_path / "manifest.json"

    # 1. Unduh dari R2 jika folder/manifest belum ada
    if not chroma_path.exists() or not manifest_path.exists():
        print(f"Indeks lokal tidak ditemukan di {chroma_path}. Mengunduh dari R2...")

        if not all([bucket, account_id, access_key, secret_key]):
            raise ValueError("Kredensial R2 tidak lengkap di Environment Variables!")

        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_key = s3_key[len(prefix) :]
                download_path = chroma_path / relative_key

                download_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, s3_key, str(download_path))
                print(f"Mengunduh: {relative_key}")

        print("Unduhan korpus selesai.")

    # 2. Validasi kelayakan dan kompatibilitas Manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan di {manifest_path} setelah proses inisialisasi."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    installed_chroma_version = version("chromadb")

    if manifest.get("chromadb_version") != installed_chroma_version:
        raise RuntimeError(
            f"FATAL: Versi ChromaDB tidak cocok!\n"
            f"Versi di korpus: {manifest.get('chromadb_version')}\n"
            f"Versi terpasang: {installed_chroma_version}\n"
            f"Pastikan versi di requirements.txt di-pin dengan benar."
        )

    if manifest.get("embed_model") != expected_embed_model:
        raise RuntimeError(
            f"FATAL: Model Embedding tidak cocok!\n"
            f"Model di korpus: {manifest.get('embed_model')}\n"
            f"Model diharapkan: {expected_embed_model}"
        )

    print(
        f"Korpus valid! Versi: {manifest.get('corpus_version')} | Chunk: {manifest.get('chunk_count')}"
    )


# ==========================================
# CONTOH PEMAKAIAN DI FASTAPI (Lifespan)
# ==========================================
# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# from src.infrastructure.advisor_chat.loader import ensure_corpus
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     ensure_corpus()
#     yield
#
# app = FastAPI(lifespan=lifespan)#
# app = FastAPI(lifespan=lifespan)
