import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.embed_corpus import load_from_file
from scripts.embed.local_embedder import LocalEmbedder

# Build a tiny corpus file: just AAPL 2023 chunks (small, no giant tables).
src = Path("data/chunks/corpus.jsonl")
aapl = [json.loads(l) for l in src.open()
        if l.strip() and json.loads(l)["ticker"] == "AAPL"
        and json.loads(l)["fiscal_year"] == 2023]
tiny = Path("data/chunks/_test_corpus.jsonl")
tiny.write_text("".join(json.dumps(c) + "\n" for c in aapl))
print(f"tiny corpus: {len(aapl)} AAPL 2023 chunks")

# Embed it to a test embedded-file (local embedder, small so no OOM).
embedder = LocalEmbedder()
out = Path("data/chunks/_test_embedded.jsonl")
written = 0
with out.open("w") as fh:
    texts = [c["text"] for c in aapl]
    vectors = embedder.embed(texts, show_progress=True)
    for c, v in zip(aapl, vectors):
        c["embedding"] = v
        fh.write(json.dumps(c) + "\n")
        written += 1
print(f"embedded {written} chunks to test file")

# Now the real test: load_from_file into the store.
loaded = load_from_file(str(out), force=False, verbose=True)
print(f"\nload_from_file returned: {loaded}")

# Verify in the DB.
import psycopg
from scripts.store.db import get_dsn
with psycopg.connect(get_dsn()) as conn, conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
    print("rows with embeddings in store:", cur.fetchone()[0])

# Cleanup test files.
tiny.unlink(); out.unlink()
print("cleaned up test files")
