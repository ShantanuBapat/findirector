import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import extract_primary_document, identify_section

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

# Scan EVERY element (not just TopSectionTitle) whose text starts with an Item
# heading, and show what type sec-parser assigned it.
print("All elements whose text looks like an Item heading:\n")
for i, e in enumerate(elements):
    text = e.text.replace("\xa0", " ").strip()
    section = identify_section(text)
    # Only show short, heading-like matches (avoid paragraphs that mention 'item')
    if section is not None and len(text) < 90:
        print(f"  [{i:3}] {type(e).__name__:20} {section:22} {text[:55]!r}")
