# Plan: pgvector → ChromaDB (service terpisah) di `ai/`

Konteks: temanmu mengembangkan `advisor_chat` dengan ChromaDB sebagai service sendiri. Dokumen ini memetakan apa yang berubah di `ai/` dan `backend/` kalau penyimpanan RAG pindah dari Postgres+pgvector ke Chroma.

## 1. Ringkasan dampak

Perombakan **sedang, terlokalisir**: hanya 3 file infrastruktur + 1 titik panggil di service. Domain (chunking, intent, guardrail, facts, schemas) dan LLM/prompt tidak berubah. Risiko terbesar bukan di kode, tapi di **kualitas retrieval** (hybrid search hilang) dan **sinkronisasi metadata dokumen**.

| Area | File | Dampak |
|---|---|---|
| Koneksi/penyimpanan | `src/infrastructure/advisor_chat/store.py` | Tulis ulang (psycopg → chromadb client) |
| Pencarian | `src/infrastructure/advisor_chat/retrieval.py` | Tulis ulang bagian `candidates`, `known_tickers`, `load`. `rrf`, `or_query`, `dedupe_language`, rerank tetap |
| Ingest | `src/infrastructure/advisor_chat/ingest.py` | Ubah pemanggilan `store.*`; alur R2 → extract → chunk → embed tetap |
| Titik panggil | `src/features/advisor_chat/service.py` (`_search`) | Ganti `store.connect()` ke client Chroma |
| Config | `src/core/config.py` | Tambah `chroma_host/port/token/collection`; `database_url` tidak dipakai RAG lagi |
| Dependensi | `pyproject.toml` (extra `rag`) | Hapus `pgvector`, `psycopg`; tambah `chromadb-client` |
| Schema DB | `backend/db/migrations/20261020000001_create_rag.sql` | Tidak dipakai lagi (lihat §5) |
| Test | `tests/test_rag.py` | `rrf`/`or_query`/intent/chunking/guardrail tetap; test yang mem-patch `retrieval` tetap jalan; tambah test store/retrieval baru |
| Infra | `docker-compose.yml`, Railway | Tambah service Chroma + volume |

Tidak berubah: `embedder.py` (bge-m3 + reranker), `r2.py`, `extract.py`, `chunking.py`, `intent.py`, `guardrail.py`, `facts.py`, `openrouter.py`, endpoint API (kontrak ke backend tetap).

## 2. Keputusan yang harus diambil dulu (tanya temanmu)

1. **Embedding dihitung di mana?** Rekomendasi: tetap di `ai/` (bge-m3, 1024 dim), kirim vektor ke Chroma lewat `embeddings=`. Jangan pakai embedding function bawaan Chroma, supaya model query = model ingest dan reranker tidak berubah.
2. **Hybrid search (dense + sparse)**: sekarang dense (pgvector) + sparse (tsvector) digabung RRF. Chroma lokal = dense saja. Pilihan:
   - **A (rekomendasi, murah):** dense-only dari Chroma + rerank cross-encoder yang sudah ada. Sparse dilepas. Uji dulu dampaknya.
   - **B:** BM25 sendiri di sisi `ai/` (mis. `rank_bm25`) atas corpus di memori. Menambah RAM dan waktu startup.
   - **C:** pakai fitur sparse/hybrid Chroma kalau versi yang dipakai mendukung (belum saya verifikasi).
3. **Metadata dokumen** (`rag_documents`: status active/superseded, etag, scanned, pages): taruh di metadata chunk di Chroma, atau tetap di Postgres? Rekomendasi: **metadata chunk di Chroma** (`source_type`, `ticker`, `fiscal_year`, `language`, `status`, `r2_key`, `etag`, `page_start`, `section`), tanpa tabel lagi.
4. **Siapa pemilik data Chroma di production?** Satu service Chroma di Railway + volume, atau Chroma Cloud. Menentukan env var di §6.

## 3. Pemetaan fungsi lama → baru

| Sekarang (Postgres) | Chroma |
|---|---|
| `store.connect()` | `chromadb.HttpClient(host, port, headers={"Authorization": ...})`, satu collection `rag_chunks`, `metadata={"hnsw:space": "cosine"}` |
| `known_etags(prefix)` | `collection.get(where={...}, include=["metadatas"])` lalu ambil `r2_key`→`etag`. Filter prefix dilakukan di Python (Chroma tidak punya LIKE) |
| `save_document` (DELETE + INSERT transaksi) | `collection.delete(where={"r2_key": key})` lalu `collection.add(ids, embeddings, documents, metadatas)`. **Tidak atomik**, lihat §4 |
| `known_tickers` (`SELECT DISTINCT`) | Tidak ada DISTINCT. Pilihan: cache set ticker di memori (refresh saat ingest/TTL), atau simpan dokumen "registry" khusus |
| `candidates` dense (`<=>` + filter SQL) | `collection.query(query_embeddings=[vec], n_results=k, where={"$and":[status active, source_type $in, ticker $in]})` |
| `candidates` sparse (`tsv @@`) | Lihat keputusan §2.2 |
| `load(ids)` (join dokumen) | `query(..., include=["documents","metadatas"])` sudah membawa teks+metadata, jadi `load` hilang (satu round trip) |
| id chunk `bigint` | id string deterministik: `f"{r2_key}#{ordinal}"` (idempoten, aman untuk re-ingest). `rrf` generik terhadap id, jadi tipe `list[int]` perlu diganti `list[str]` |

## 4. Risiko dan hal yang hilang

- **Atomisitas ingest**: Postgres mengganti dokumen dalam satu transaksi, Chroma tidak. Mitigasi: `add` dulu dengan id baru, baru hapus yang lama; atau tulis `etag` ke metadata terakhir dan anggap dokumen "belum selesai" bila jumlah chunk tidak cocok. Ingest sudah resumable lewat etag, jadi cukup tambah cek jumlah chunk.
- **Filter ticker/source_type** wajib dites: `$in` dengan list kosong/`None` harus dihindari (bangun `where` hanya dari filter yang ada).
- **Batas `n_results` + filter**: Chroma menerapkan filter metadata saat pencarian; hasil bisa kurang dari `k`. Aman, tapi `rag_top_k=50` perlu dites terhadap rerank.
- **Skala**: jumlah chunk korpus belum saya hitung. Chroma single-node cukup untuk puluhan-ratusan ribu chunk; bila lebih, ukur dulu.
- **Dua sumber kebenaran**: kalau metadata tetap di Postgres dan vektor di Chroma, keduanya bisa menyimpang. Karena itu rekomendasi §2.3.
- **Auth Chroma**: service Chroma harus private (jaringan internal Railway) atau pakai token. Jangan buka publik.
- **Versi klien vs server Chroma** harus cocok; pin keduanya.

## 5. Dampak ke `backend/`

- Migrasi `20261020000001_create_rag.sql` dan ekstensi `vector` tidak lagi dibutuhkan oleh `ai/`. **Jangan hapus migrasi yang sudah dijalankan di DB mana pun**; tambah migrasi baru `DROP TABLE rag_chunks, rag_documents` hanya setelah Chroma terbukti jalan di production.
- Konsekuensi positif: Postgres production bisa pakai Postgres biasa (tidak perlu template pgvector), kecuali `pg_trgm` yang masih dipakai migrasi market (extension bawaan Postgres standar).
- `backend` tidak memanggil Chroma; kontrak `/internal` ke `ai/` tidak berubah.

## 6. Perubahan infra/env

- `docker-compose.yml`: tambah service `chroma` (image `chromadb/chroma`, versi di-pin), volume untuk persist, healthcheck; `ai` `depends_on` chroma.
- Railway: service baru Chroma + volume, bind jaringan privat (`chroma.railway.internal`), `ai` memakai alamat itu.
- Env baru di `ai`: `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_TOKEN` (bila auth), `CHROMA_COLLECTION=rag_chunks`.
- Env lama yang tidak dipakai RAG lagi: `DATABASE_URL` di `ai` (cek dulu apakah ada fitur lain yang memakainya; dari pembacaan saya hanya `store.py`).
- `ai/railway.json` dan healthcheck `/health` tetap.

## 7. Urutan kerja

1. Sepakati keputusan §2 dengan temanmu (terutama hybrid dan lokasi metadata).
2. Tambah config + dependensi + service Chroma di compose (belum menyentuh logika).
3. Tulis `store.py` baru (client, add/delete, etag). Unit test dengan Chroma in-memory (`chromadb.EphemeralClient`).
4. Tulis `retrieval.candidates/known_tickers` baru; hapus `load`; ganti tipe id ke string; sesuaikan `rrf`.
5. Ubah `ingest.py` dan `service._search`; hapus pemakaian psycopg/pgvector.
6. Re-ingest korpus dari R2 ke Chroma (`python -m src.infrastructure.advisor_chat.ingest --prefix ... --force`). Embedding ulang tidak perlu bila model sama, tapi pipeline tidak menyimpan vektor di luar DB, jadi tetap perlu jalan penuh.
7. **Evaluasi retrieval** sebelum cutover: kumpulkan ±20–30 pertanyaan nyata (regulasi, laporan emiten, ticker spesifik), bandingkan top-6 lama vs baru; keputusan §2.2 diambil dari hasil ini.
8. Cutover di staging, lalu production; baru kemudian migrasi drop tabel RAG (§5) dan hapus dependensi lama.

## 8. Definisi selesai

- `pytest` hijau (test lama non-DB tetap, test store/retrieval baru lewat Chroma in-memory).
- Pertanyaan evaluasi memberi sitasi sebanding dengan sebelum migrasi.
- Ingest ulang dokumen yang sama tidak membuat duplikat.
- `ai` start dan `/health` ok tanpa `DATABASE_URL`.

## 9. Belum diverifikasi

Ukuran korpus, dukungan sparse/hybrid di versi Chroma yang akan dipakai, apakah ada kode di luar `ai/src` yang membaca tabel `rag_*` langsung (saya hanya mencari di `ai/`), dan perilaku `where` Chroma pada filter gabungan besar.
