import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder
from scripts.store.pgvector_store import PgVectorStore
from scripts.retrieval.retrieve import retrieve

embedder = LocalEmbedder()
store = PgVectorStore()

query = "What was Apple's research and development spending in fiscal 2023?"
result = retrieve(query, {"company": "AAPL", "year": 2023}, embedder, store, k=5)

for i, c in enumerate(result["chunks"], 1):
    preview = c["text"][:200].replace("\n", " ")
    has_rd = "research and development" in c["text"].lower() or "29915" in c["text"]
    print(f"[{i}] {c['section']} (score {c['score']:.3f}) has_R&D={has_rd}")
    print(f"    {preview}\n")

# Does the R&D chunk even exist for AAPL 2023, and what's its rank if we widen k?
print("=" * 60)
print("Searching AAPL 2023 with k=20 to find the R&D chunk's rank:")
qvec = embedder.embed([query])[0]
wide = store.search(qvec, k=20, filters={"ticker": "AAPL", "fiscal_year": 2023})
for i, c in enumerate(wide, 1):
    if "29915" in c["text"] or "research and development" in c["text"].lower():
        print(f"  R&D chunk found at rank {i} (score {c['score']:.3f}, section {c['section']})")
