from unittest.mock import MagicMock

import pytest

from src.infrastructure.advisor_chat import embedder, retrieval


def mock_embed(texts):
    return [[0.1] * 1024]


def test_retrieval_empty_results(monkeypatch):
    monkeypatch.setattr(embedder, "embed", mock_embed)

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [[]], "metadatas": [[]]}
    monkeypatch.setattr(retrieval, "get_collection", lambda: mock_col)

    results = retrieval.candidates("query kosong", types=None, tickers=None, k=50)
    assert isinstance(results, list)
    assert len(results) == 0


def test_candidates_filters(monkeypatch):
    monkeypatch.setattr(embedder, "embed", mock_embed)

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [[]], "metadatas": [[]]}
    monkeypatch.setattr(retrieval, "get_collection", lambda: mock_col)

    retrieval.candidates("test query", types=None, tickers=None, k=50)

    call_args = mock_col.query.call_args[1]
    where = call_args["where"]

    assert where == {
        "$and": [{"status": "approved"}, {"expiry_date": {"$gte": retrieval.today_int()}}]
    }


def test_passage_metadata_mapping(monkeypatch):
    monkeypatch.setattr(embedder, "embed", mock_embed)

    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["Isi teks pasal"]],
        "metadatas": [
            [
                {
                    "doc_id": "doc_123",
                    "title": "POJK 1",
                    "doc_type": "regulasi",
                    "page": 2,
                    "section": "Pasal 1",
                    "r2_key": "ojk/pojk1.pdf",
                    "tier": 1,
                    "nomor": "1/POJK/2026",
                    "publisher": "OJK",
                    "published_date": 20260101,
                }
            ]
        ],
    }
    monkeypatch.setattr(retrieval, "get_collection", lambda: mock_col)

    results = retrieval.candidates("test query", types=None, tickers=None, k=50)

    assert len(results) == 1
    p = results[0]
    assert p.id == "doc_123"
    assert p.source_type == "ojk_regulasi"
    assert p.section == "Pasal 1"
    assert p.tier == 1
    assert p.nomor == "1/POJK/2026"
    assert p.tanggal == 20260101
