import warnings
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import extract_primary_document, walk_sections

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

metadata = {"ticker": "AAPL", "fiscal_year": 2025, "filing_type": "10-K"}
records = walk_sections(elements, metadata)

print(f"{len(elements)} elements -> {len(records)} records\n")

# Breakdown by content type
print("by content_type:", dict(Counter(r["content_type"] for r in records)))
print()

# Records per section (in first-seen order)
print("records per section:")
seen = []
for r in records:
    if r["section"] not in seen:
        seen.append(r["section"])
for section in seen:
    rs = [r for r in records if r["section"] == section]
    n_narr = sum(1 for r in rs if r["content_type"] == "narrative")
    n_tbl = sum(1 for r in rs if r["content_type"] == "table")
    print(f"  {str(section):26} narrative={n_narr}  table={n_tbl}")

# Spot-check: does MD&A (the critical section) have content?
print()
mdna = [r for r in records if r["section"] == "mdna"]
print(f"mdna records: {len(mdna)}")
if mdna:
    sample = next((r for r in mdna if r["content_type"] == "narrative"), mdna[0])
    print("first mdna text (200 chars):")
    print(" ", repr(sample["text"][:200]))
