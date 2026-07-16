import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder
from scripts.store.pgvector_store import PgVectorStore
from scripts.retrieval.retrieve import retrieve

embedder = LocalEmbedder()
store = PgVectorStore()

def show(label, query, params):
    print(f"\n{'='*70}\n{label}\nQUERY: {query}\nPARAMS: {params}")
    result = retrieve(query, params, embedder, store, k=3)
    print(f"STATUS: {result['status']}")
    if result["status"] == "ok":
        for i, c in enumerate(result["chunks"], 1):
            print(f"  [{i}] {c['ticker']} {c['fiscal_year']} {c['section']} "
                  f"(score {c['score']:.3f})")
    else:
        print(f"  reason: {result['reason']} | filters: {result['filters']}")

# 1. Normal hit — company + year in corpus
show("1. HIT (AAPL 2023)",
     "What was research and development spending?",
     {"company": "AAPL", "year": 2023})

# 2. Berkshire — proves ticker normalization (BRK.B -> BRK-B)
show("2. NORMALIZATION (BRK.B -> BRK-B)",
     "How does the company describe insurance float?",
     {"company": "BRK.B", "year": 2023})

# 3. Null year — filters by ticker only, ranks across all years
show("3. NULL YEAR (ticker-only filter)",
     "What was Johnson & Johnson's R&D spending?",
     {"company": "JNJ", "year": None})

# 4. Corpus-boundary decline — company not in corpus
show("4. DECLINE (NFLX not in corpus)",
     "What were the revenues?",
     {"company": "NFLX", "year": 2023})
