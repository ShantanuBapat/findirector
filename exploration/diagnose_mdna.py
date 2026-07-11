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

# Find the index range between the Item 7 and Item 8 headings.
start = end = None
for i, e in enumerate(elements):
    t = e.text.replace("\xa0", " ").strip().lower()
    if start is None and t.startswith("item 7.") and "management" in t:
        start = i
    elif start is not None and t.startswith("item 8."):
        end = i
        break

print(f"Item 7 heading at [{start}], Item 8 heading at [{end}]")
print(f"{end - start - 1} elements inside MD&A\n")

for i in range(start, end + 1):
    e = elements[i]
    t = e.text.replace("\xa0", " ").strip()
    sec = identify_section(t)
    flag = ""
    if sec is not None:
        flag = f"  <-- identify_section = {sec}"
    print(f"  [{i:3}] {type(e).__name__:18} {t[:60]!r}{flag}")
