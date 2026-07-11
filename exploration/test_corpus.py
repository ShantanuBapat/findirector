import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chunk_filings import chunk_corpus, TARGET_TOKENS

ROOT = "data/raw/sec_edgar/sec-edgar-filings"

chunks, errors = chunk_corpus(ROOT)

print("\n" + "=" * 60)
print(f"total chunks: {len(chunks)}")
print(f"errors: {len(errors)}")
for path, err in errors:
    print("  ", path, err)

# Content-type split
print("\nby content_type:", dict(Counter(c["content_type"] for c in chunks)))

# Cross-company splitter check: any NARRATIVE chunk over the token limit?
narr_over = [c for c in chunks
             if c["content_type"] == "narrative" and c["n_tokens"] > TARGET_TOKENS]
print(f"\nnarrative chunks over {TARGET_TOKENS} tokens: {len(narr_over)}  (should be 0)")
for c in narr_over[:10]:
    print(f"  {c['ticker']} {c['fiscal_year']} {c['section']}: {c['n_tokens']} tok")

# Tables over limit (expected/atomic) - just count
tbl_over = [c for c in chunks
            if c["content_type"] == "table" and c["n_tokens"] > TARGET_TOKENS]
print(f"table chunks over {TARGET_TOKENS} (atomic, expected): {len(tbl_over)}")

# Chunks per ticker (should be roughly even; wild outliers = a parse problem)
per_ticker = Counter(c["ticker"] for c in chunks)
print("\nchunks per ticker:")
for t in sorted(per_ticker):
    print(f"  {t:6} {per_ticker[t]}")

# Section coverage: how many distinct sections got chunks, and any None?
sections = Counter(c["section"] for c in chunks)
print(f"\ndistinct sections: {len(sections)}")
none_ct = sections.get(None, 0)
print(f"chunks with section=None (pre-Item-1 content): {none_ct}")
