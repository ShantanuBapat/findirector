import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import (
    extract_primary_document, extract_filing_metadata, walk_sections,
    TARGET_TOKENS, _count_tokens, _split_oversized, _SENTENCE_RE,
)

root = Path("data/raw/sec_edgar/sec-edgar-filings")
f = next(p for p in sorted((root / "JNJ" / "10-K").glob("*/full-submission.txt"))
         if extract_filing_metadata(p.read_text(), "JNJ")["fiscal_year"] == 2023)
text = f.read_text()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(extract_primary_document(text))
records = walk_sections(elements, extract_filing_metadata(text, "JNJ"))
fin = next(r for r in records if r["section"] == "financials" and r["content_type"] == "narrative")

for p in [x for x in fin["text"].split("\n\n") if x.strip()]:
    if _count_tokens(p) <= TARGET_TOKENS:
        continue
    pieces = _split_oversized(p)
    for piece in pieces:
        if _count_tokens(piece) > TARGET_TOKENS:
            # The piece is a join of sentences. Compare sum-of-parts vs whole.
            sents = _SENTENCE_RE.split(piece)
            sum_parts = sum(_count_tokens(s) for s in sents)
            whole = _count_tokens(piece)
            print(f"piece: {len(sents)} sentences")
            print(f"  sum of per-sentence token counts: {sum_parts}")
            print(f"  token count of the joined whole:   {whole}")
            print(f"  additivity gap (whole - sum):      {whole - sum_parts}")
