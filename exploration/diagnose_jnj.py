import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, ticker_from_path, extract_filing_metadata,
    walk_sections, TARGET_TOKENS, OVERLAP_TOKENS, _count_tokens,
    _split_oversized, _pack_paragraphs,
)

root = Path("data/raw/sec_edgar/sec-edgar-filings")
# JNJ 2023 filing (period of report year 2023)
jnj_files = sorted((root / "JNJ" / "10-K").glob("*/full-submission.txt"))
target = None
for f in jnj_files:
    text = f.read_text()
    meta = extract_filing_metadata(text, "JNJ")
    if meta["fiscal_year"] == 2023:
        target = f
        break

text = target.read_text()
html = extract_primary_document(text)
meta = extract_filing_metadata(text, "JNJ")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)
records = walk_sections(elements, meta)

fin = next(r for r in records
          if r["section"] == "financials" and r["content_type"] == "narrative")
print(f"financials narrative blob: {_count_tokens(fin['text'])} tokens")

# Rebuild the units the packer sees
paras = [p for p in fin["text"].split("\n\n") if p.strip()]
units = []
for p in paras:
    if _count_tokens(p) > TARGET_TOKENS:
        units.extend(_split_oversized(p))
    else:
        units.append(p)

sizes = [_count_tokens(u) for u in units]
print(f"{len(units)} units")
print("units in 416..500 range:", sorted([s for s in sizes if 416 < s <= 500], reverse=True))
print("units EXACTLY over 480:", sorted([s for s in sizes if s > 480], reverse=True))

# Now find the 481 chunk and show what it's made of
chunks = _pack_paragraphs(fin["text"])
for i, c in enumerate(chunks):
    n = _count_tokens(c)
    if n > TARGET_TOKENS:
        sub = [_count_tokens(p) for p in c.split("\n\n")]
        print(f"\nOVER chunk {i}: {n} tok, composed of paragraph sizes {sub}")
        print("  is it a single unit?", len(sub) == 1)
