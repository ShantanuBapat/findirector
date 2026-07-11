import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import extract_primary_document

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

TITLE_TYPES = (sp.TopSectionTitle, sp.TitleElement)

# For every element whose text STARTS with "Item N.", show its type.
import re
anchored = re.compile(r"^\s*item\s+\d+[a-c]?\s*[.\-]", re.IGNORECASE)
print("Elements whose text starts with an 'Item N.' heading pattern:\n")
for i, e in enumerate(elements):
    t = e.text.replace("\xa0", " ").strip()
    if anchored.match(t):
        is_title = isinstance(e, TITLE_TYPES)
        mark = "" if is_title else "   <-- NOT a title type!"
        print(f"  [{i:3}] {type(e).__name__:18} len={len(t):4} {t[:45]!r}{mark}")
