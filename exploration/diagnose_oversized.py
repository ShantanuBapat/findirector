import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, walk_sections, TARGET_TOKENS,
    _count_tokens, _split_oversized, _SENTENCE_RE,
)

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

records = walk_sections(elements, {"ticker": "AAPL", "fiscal_year": 2025, "filing_type": "10-K"})
rf = next(r for r in records if r["section"] == "risk_factors" and r["content_type"] == "narrative")
paras = [p for p in rf["text"].split("\n\n") if p.strip()]

# The big paragraphs that go through _split_oversized
big = [p for p in paras if _count_tokens(p) > TARGET_TOKENS]
print(f"{len(big)} paragraphs exceed TARGET ({TARGET_TOKENS}):",
      [_count_tokens(p) for p in big], "\n")

for p in big:
    print(f"paragraph of {_count_tokens(p)} tokens:")
    sents = _SENTENCE_RE.split(p)
    print(f"  splits into {len(sents)} 'sentences', sizes: {[_count_tokens(s) for s in sents][:20]}")
    pieces = _split_oversized(p)
    print(f"  _split_oversized -> {len(pieces)} pieces, sizes: {[_count_tokens(x) for x in pieces]}")
    over = [x for x in pieces if _count_tokens(x) > TARGET_TOKENS]
    print(f"  pieces OVER target: {len(over)}\n")
