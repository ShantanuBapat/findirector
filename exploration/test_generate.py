import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder
from scripts.store.pgvector_store import PgVectorStore
from scripts.retrieval.retrieve import retrieve
from scripts.generation.anthropic_generator import AnthropicGenerator

embedder = LocalEmbedder()
store = PgVectorStore()
generator = AnthropicGenerator()

query = "What was Apple's research and development spending in fiscal 2023?"
params = {"company": "AAPL", "year": 2023, "fact_requested": "research and development spending"}

print(f"QUERY: {query}\nPARAMS: {params}\n")

result = retrieve(query, params, embedder, store, k=5)
print(f"retrieval status: {result['status']}")

if result["status"] == "ok":
    print(f"retrieved {len(result['chunks'])} chunks\n")
    answer = generator.generate(query, result["chunks"])
    print("=" * 70)
    print("ANSWER:")
    print(answer)
    print("=" * 70)
else:
    print("declined:", result["reason"])
