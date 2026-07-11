import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, walk_sections, TARGET_TOKENS, OVERLAP_TOKENS,
    _count_tokens, _pack_paragraphs,
)

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

records = walk_sections(elements, {"ticker": "AAPL", "fiscal_year": 2025, "filing_type": "10-K"})
rf = next(r for r in records if r["section"] == "risk_factors" and r["content_type"] == "narrative")

print(f"TARGET={TARGET_TOKENS} OVERLAP={OVERLAP_TOKENS}")
print(f"source risk_factors narrative: {_count_tokens(rf['text'])} tokens\n")

# Show the paragraph sizes feeding the packer
paras = [p for p in rf["text"].split("\n\n") if p.strip()]
print(f"{len(paras)} paragraphs; token sizes:")
print(" ", [_count_tokens(p) for p in paras][:30])
print()

chunks = _pack_paragraphs(rf["text"])
print(f"{len(chunks)} chunks, token sizes:")
for i, c in enumerate(chunks):
    flag = "  <-- OVER" if _count_tokens(c) > TARGET_TOKENS else ""
    print(f"  chunk {i}: {_count_tokens(c)} tok{flag}")
