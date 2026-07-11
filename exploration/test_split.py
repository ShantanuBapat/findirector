import warnings
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, walk_sections, split_records, TARGET_TOKENS,
)

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

metadata = {"ticker": "AAPL", "fiscal_year": 2025, "filing_type": "10-K"}
records = walk_sections(elements, metadata)
chunks = split_records(records)

print(f"{len(records)} records -> {len(chunks)} chunks")
print("by content_type:", dict(Counter(c["content_type"] for c in chunks)))
print()

# Token distribution
toks = [c["n_tokens"] for c in chunks]
narr = [c["n_tokens"] for c in chunks if c["content_type"] == "narrative"]
print(f"tokens: min={min(toks)} max={max(toks)} mean={sum(toks)//len(toks)}")
over = [c for c in chunks if c["n_tokens"] > TARGET_TOKENS]
print(f"chunks over TARGET_TOKENS ({TARGET_TOKENS}): {len(over)}")
for c in over[:8]:
    print(f"  {c['section']}/{c['content_type']}  {c['n_tokens']} tok")
print()

# MD&A: the section that was broken — how did it split?
mdna = [c for c in chunks if c["section"] == "mdna"]
print(f"mdna chunks: {len(mdna)} "
      f"(narrative={sum(1 for c in mdna if c['content_type']=='narrative')}, "
      f"table={sum(1 for c in mdna if c['content_type']=='table')})")
print("mdna narrative token sizes:",
      [c["n_tokens"] for c in mdna if c["content_type"] == "narrative"])
