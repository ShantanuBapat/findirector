import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder

print("Loading BGE-M3 (first run downloads ~2.3GB)...")
embedder = LocalEmbedder()
print(f"loaded on device: {embedder.device}")
print(f"reported dimension: {embedder.dimension}")

# Embed a single test sentence
text = "What was Apple's research and development spending in fiscal 2023?"
vectors = embedder.embed([text])

print(f"\nembedded 1 text -> {len(vectors)} vector(s)")
print(f"vector dimension: {len(vectors[0])}")
print(f"first 5 values: {vectors[0][:5]}")

# Verify normalization: a unit vector's L2 norm should be ~1.0
import math
norm = math.sqrt(sum(x * x for x in vectors[0]))
print(f"L2 norm (should be ~1.0): {norm:.4f}")

# Sanity: dimension must match what the store expects
assert len(vectors[0]) == embedder.dimension == 1024, "dimension mismatch!"
print("\nOK: 1024-dim normalized vector, matches store column")
