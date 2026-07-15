import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder
from scripts.store.pgvector_store import PgVectorStore

embedder = LocalEmbedder()
store = PgVectorStore()

def run_query(question, k=3, filters=None):
    print(f"\n{'='*70}\nQUERY: {question}")
    if filters:
        print(f"FILTERS: {filters}")
    qvec = embedder.embed([question])[0]
    results = store.search(qvec, k=k, filters=filters)
    print(f"-> {len(results)} results")
    for i, r in enumerate(results, 1):
        preview = r["text"][:150].replace("\n", " ")
        print(f"  [{i}] {r['ticker']} {r['fiscal_year']} {r['section']} "
              f"(score {r['score']:.3f})")
        print(f"      {preview}...")

# 1. Unfiltered semantic search
run_query("What are the main risks the company faces?", k=3)

# 2. Filtered to a specific company + year (the directive-driven pattern)
run_query("What was research and development spending?",
          k=3, filters={"ticker": "AAPL", "fiscal_year": 2023})

# 3. Corpus-boundary guard: filter to a company NOT in the corpus
run_query("What were the revenues?",
          k=3, filters={"ticker": "NFLX", "fiscal_year": 2023})
print("  ^ empty result = corpus-boundary guard signal (decline)")
