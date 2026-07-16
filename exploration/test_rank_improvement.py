import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder
from scripts.store.pgvector_store import PgVectorStore
from scripts.retrieval.retrieve import _embedding_text

embedder = LocalEmbedder()
store = PgVectorStore()

query = "What was Apple's research and development spending in fiscal 2023?"

# OLD behavior: embed the full query
old_vec = embedder.embed([query])[0]
old = store.search(old_vec, k=20, filters={"ticker": "AAPL", "fiscal_year": 2023})

# NEW behavior: embed the distilled fact_requested
params = {"company": "AAPL", "year": 2023, "fact_requested": "R&D spending"}
new_text = _embedding_text(query, params)
new_vec = embedder.embed([new_text])[0]
new = store.search(new_vec, k=20, filters={"ticker": "AAPL", "fiscal_year": 2023})

def rd_rank(results):
    for i, c in enumerate(results, 1):
        if "29915" in c["text"] or "research and development" in c["text"].lower():
            return i, c["score"]
    return None, None

print(f"OLD (embed full query): R&D chunk rank = {rd_rank(old)}")
print(f"NEW (embed '{new_text}'): R&D chunk rank = {rd_rank(new)}")
