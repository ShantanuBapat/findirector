import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, walk_sections, TARGET_TOKENS, OVERLAP_TOKENS,
    _count_tokens, _split_oversized,
)

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

records = walk_sections(elements, {"ticker": "AAPL", "fiscal_year": 2025, "filing_type": "10-K"})
rf = next(r for r in records if r["section"] == "risk_factors" and r["content_type"] == "narrative")
paras = [p for p in rf["text"].split("\n\n") if p.strip()]

# Rebuild the units list the way _pack_paragraphs does
units = []
for p in paras:
    if _count_tokens(p) > TARGET_TOKENS:
        units.extend(_split_oversized(p))
    else:
        units.append(p)

budget = TARGET_TOKENS - OVERLAP_TOKENS
sizes = [_count_tokens(u) for u in units]
danger = [s for s in sizes if budget < s <= TARGET_TOKENS]
print(f"content_budget={budget}, TARGET={TARGET_TOKENS}")
print(f"{len(units)} units; sizes over budget (>{budget}): "
      f"{sorted([s for s in sizes if s > budget], reverse=True)}")
print(f"units in danger band ({budget}<s<={TARGET_TOKENS}): {sorted(danger, reverse=True)}")
