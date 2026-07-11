import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, extract_filing_metadata, walk_sections,
    TARGET_TOKENS, _count_tokens, _split_oversized, _SENTENCE_RE, _ENCODING,
)

root = Path("data/raw/sec_edgar/sec-edgar-filings")
f = next(p for p in sorted((root / "JNJ" / "10-K").glob("*/full-submission.txt"))
         if extract_filing_metadata(p.read_text(), "JNJ")["fiscal_year"] == 2023)
text = f.read_text()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(extract_primary_document(text))

records = walk_sections(elements, extract_filing_metadata(text, "JNJ"))
fin = next(r for r in records
           if r["section"] == "financials" and r["content_type"] == "narrative")

paras = [p for p in fin["text"].split("\n\n") if p.strip()]
for p in paras:
    if _count_tokens(p) <= TARGET_TOKENS:
        continue
    pieces = _split_oversized(p)
    for piece in pieces:
        n = _count_tokens(piece)
        if n > TARGET_TOKENS:
            sents = _SENTENCE_RE.split(p)
            big = [s for s in sents if _count_tokens(s) > TARGET_TOKENS]
            ids = _ENCODING.encode(piece)
            print("481-producing paragraph:", _count_tokens(p), "tok")
            print("  sentences:", len(sents), "| sentences over TARGET:", len(big))
            print("  offending piece:", n, "tok | encodes to", len(ids), "ids")
