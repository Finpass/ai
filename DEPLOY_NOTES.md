Set-Content -Path "DEPLOY_NOTES.md" -Encoding UTF8 -Value @"
# Deployment Notes: RAG Module (ChromaDB)

## 1. Environment Variables (Railway)
Pastikan variable berikut diset di dashboard Railway:
- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID` (gunakan kredensial dengan akses read-only demi keamanan)
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`
- `OPENROUTER_API_KEY`
- `EMBED_MODEL=baai/bge-m3`
- `CHROMA_PATH=chroma_db`
- `CORPUS_PREFIX=artifacts/chroma_corpus_v1/`

## 2. Requirements
Versi ChromaDB wajib di-pin agar kompatibel dengan format database SQLite yang sudah dibekukan. Sesuaikan pada dependencies/requirements:
chromadb==1.5.9

## 3. Command Start (Sangat Penting)
ChromaDB tidak mendukung akses multiproses ke SQLite. Aplikasi FastAPI wajib dijalankan hanya dengan satu worker.
Contoh: uvicorn src.app:app --host 0.0.0.0 --port `$PORT --workers 1

## 4. Perkiraan Waktu Startup Pertama
Saat deploy pertama (atau restart tanpa persistent volume), aplikasi akan mengunduh database dari R2. Dengan jaringan Railway, ini akan memakan waktu sekitar 30 - 60 detik sebelum aplikasi siap menerima request.

## 5. Verifikasi Deploy
Disarankan menambahkan informasi RAG di endpoint health check (/health) yang menampilkan total chunk (harus 62.276) dan versi korpus (v1) untuk memastikan loader berfungsi dan index terpasang dengan benar.
"@