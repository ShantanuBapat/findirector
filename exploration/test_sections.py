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

tops = [e for e in elements if type(e).__name__ == "TopSectionTitle"]
print(f"{len(tops)} TopSectionTitle elements:\n")
for e in tops:
    text = e.text.replace("\xa0", " ")[:50]
    section = identify_section(e.text)
    print(f"  {text!r:54} -> {section}")
