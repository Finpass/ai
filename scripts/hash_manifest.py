import hashlib
import json
import pathlib

root = pathlib.Path("chroma_db")
manifest_file = root / "manifest.json"

if not manifest_file.exists():
    print("Error: manifest.json tidak ditemukan!")
    exit(1)

m = json.loads(manifest_file.read_text(encoding="utf-8"))

m["sha256"] = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(root.rglob("*"))
    if p.is_file() and p.name != "manifest.json"
}

manifest_file.write_text(json.dumps(m, indent=2), encoding="utf-8")
print("Berhasil menambahkan hash SHA256 ke manifest.json")
