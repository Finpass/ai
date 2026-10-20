import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import boto3

from src.core.config import settings


def _verify(root: Path, manifest: dict) -> None:
    """Memverifikasi integritas file yang diunduh menggunakan SHA-256."""
    digests = manifest.get("sha256")
    if not digests:
        print("PERINGATAN: manifest tanpa sha256, isi korpus tidak diverifikasi.")
        return
    for rel, want in digests.items():
        h = hashlib.sha256()
        with open(root / rel, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != want:
            raise RuntimeError(f"Checksum tidak cocok: {rel}")


def ensure_corpus():
    """
    Fungsi ini dipanggil satu kali saat aplikasi FastAPI start (di lifespan).
    Mengunduh indeks ChromaDB dari R2 bila belum ada, dan memvalidasi versinya.
    """
    prefix = os.getenv("CORPUS_PREFIX", "artifacts/chroma_corpus_v2/")

    # PERBAIKAN: Menggunakan settings (config.py) sebagai satu sumber kebenaran
    bucket = settings.r2_bucket
    endpoint_url = settings.r2_endpoint
    access_key = settings.r2_access_key_id
    secret_key = settings.r2_secret_access_key

    # PERBAIKAN: Default mengarah ke settings.embed_model
    expected_embed_model = os.getenv("EMBED_MODEL", settings.embed_model)

    chroma_path = Path(settings.chroma_path)
    manifest_path = chroma_path / "manifest.json"

    downloaded = False

    # 1. Unduh dari R2 jika folder/manifest belum ada
    if not chroma_path.exists() or not manifest_path.exists():
        print(f"Indeks lokal tidak ditemukan di {chroma_path}. Mengunduh dari R2...")

        if not all([bucket, endpoint_url, access_key, secret_key]):
            raise ValueError(
                "Kredensial R2 (bucket, endpoint, access/secret key) tidak lengkap di Environment/Settings!"
            )

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_key = s3_key[len(prefix) :]

                if not relative_key:
                    continue

                download_path = chroma_path / relative_key

                download_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, s3_key, str(download_path))
                print(f"Mengunduh: {relative_key}")

        print("Unduhan korpus selesai.")
        downloaded = True

    # 2. Validasi kelayakan dan kompatibilitas Manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan di {manifest_path} setelah proses inisialisasi."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # PERBAIKAN: Hanya verifikasi hash setelah unduhan baru.
    # (Chroma mengubah file sqlite saat berjalan, jadi hash akan beda jika direstart)
    if downloaded:
        _verify(chroma_path, manifest)

    installed_chroma_version = version("chromadb")

    if manifest.get("chromadb_version") != installed_chroma_version:
        raise RuntimeError(
            f"FATAL: Versi ChromaDB tidak cocok!\n"
            f"Versi di korpus: {manifest.get('chromadb_version')}\n"
            f"Versi terpasang: {installed_chroma_version}\n"
            f"Pastikan versi di requirements.txt di-pin dengan benar."
        )

    # PERBAIKAN: Membandingkan tanpa memedulikan huruf besar/kecil (case-insensitive)
    if str(manifest.get("embed_model", "")).lower() != expected_embed_model.lower():
        raise RuntimeError(
            f"FATAL: Model Embedding tidak cocok!\n"
            f"Model di korpus: {manifest.get('embed_model')}\n"
            f"Model diharapkan: {expected_embed_model}"
        )

    print(
        f"Korpus valid! Versi: {manifest.get('corpus_version')} | Chunk: {manifest.get('chunk_count')}"
    )
