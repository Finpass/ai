from collections.abc import Iterable

import psycopg
from pgvector.psycopg import register_vector

from src.core.config import settings
from src.domain.advisor_chat.chunking import Chunk
from src.domain.advisor_chat.meta import DocMeta


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=False)
    register_vector(conn)
    return conn


def known_etags(conn: psycopg.Connection, prefix: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT r2_key, etag FROM rag_documents WHERE r2_key LIKE %s", (prefix + "%",)
    ).fetchall()
    return {k: e for k, e in rows}


def save_document(
    conn: psycopg.Connection,
    key: str,
    etag: str,
    meta: DocMeta,
    pages: int,
    scanned: bool,
    chunks: Iterable[Chunk],
    vectors: Iterable[Iterable[float]],
) -> int:
    """Replaces a document and its chunks in one transaction."""
    conn.execute("DELETE FROM rag_documents WHERE r2_key = %s", (key,))
    doc_id = conn.execute(
        """INSERT INTO rag_documents (r2_key, etag, source_type, title, ticker, fiscal_year, language, pages, scanned)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (
            key,
            etag,
            meta.source_type,
            meta.title,
            meta.ticker,
            meta.fiscal_year,
            meta.language,
            pages,
            scanned,
        ),
    ).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO rag_chunks (document_id, ordinal, page_start, page_end, section, has_table, content, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (
                    doc_id,
                    c.ordinal,
                    c.page_start,
                    c.page_end,
                    c.section,
                    c.has_table,
                    c.content,
                    list(v),
                )
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
    conn.commit()
    return doc_id
