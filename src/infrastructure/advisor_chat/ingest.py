"""Ingest PDFs from R2 into pgvector. Resumable: files whose etag is already stored are skipped.

python -m src.features.rag.ingest --prefix ojk-regulasi/ [--limit N] [--force]
"""

import argparse
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

from src.features.rag import chunking, embedder, extract, meta, r2, store

logger = logging.getLogger("rag.ingest")


def _prepare(job: tuple[str, str, str]):
    """Runs in a worker process: download, extract, chunk. The PDF is deleted right away."""
    key, source_type, title = job
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        r2.download(key, path)
        pages, scanned = extract.extract_pages(path)
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
        return key, None, 0, False, type(exc).__name__
    finally:
        if os.path.exists(path):
            os.remove(path)
    chunks = [] if scanned else chunking.chunk_pages(pages, source_type, title)
    return key, chunks, len(pages), scanned, None


def run(prefix: str, limit: int | None, force: bool, workers: int) -> None:
    conn = store.connect()
    have = {} if force else store.known_etags(conn, prefix)
    todo = []
    for f in r2.list_pdfs(prefix):
        m = meta.parse_key(f.key)
        if m is None or have.get(f.key) == f.etag:
            continue
        todo.append((f, m))
        if limit and len(todo) >= limit:
            break
    logger.info("%d documents to ingest under %r", len(todo), prefix)
    t0, chunks_total, skipped = time.time(), 0, 0
    jobs = [(f.key, m.source_type, m.title) for f, m in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (key, chunks, pages, scanned, err) in enumerate(
            pool.map(_prepare, jobs, chunksize=1), 1
        ):
            f, m = todo[i - 1]
            if err or (not chunks and not scanned):
                skipped += 1
                logger.warning("skip %s (%s)", key, err or "no text")
                if err is None:  # record empty/scanned docs so they are not retried
                    store.save_document(conn, key, f.etag, m, pages, scanned, [], [])
                continue
            if scanned:
                skipped += 1
                store.save_document(conn, key, f.etag, m, pages, True, [], [])
                continue
            vecs = embedder.embed([c.content for c in chunks])
            store.save_document(conn, key, f.etag, m, pages, False, chunks, vecs)
            chunks_total += len(chunks)
            if i % 10 == 0 or i == len(todo):
                rate = chunks_total / max(time.time() - t0, 1)
                logger.info(
                    "%d/%d docs, %d chunks, %.1f chunks/s, %d skipped",
                    i,
                    len(todo),
                    chunks_total,
                    rate,
                    skipped,
                )
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.prefix, a.limit, a.force, a.workers)
