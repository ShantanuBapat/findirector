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

# Find ANY element mentioning "Item 7" or "Item 5" or "Item 12" as a heading,
# no length filter this time.
targets = ["item 7.", "item 5.", "item 12."]
print("Searching for MD&A (Item 7), Item 5, Item 12 headings:\n")
for i, e in enumerate(elements):
    text = e.text.replace("\xa0", " ").strip()
    low = text.lower()
    for t in targets:
        if low.startswith(t):
            print(f"  [{i:3}] {type(e).__name__:16} {text[:70]!r}")
