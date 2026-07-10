import warnings
import sys
from pathlib import Path

# Make the repo root importable regardless of where python is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sec_parser as sp
from scripts.chunk_filings import extract_primary_document

p = "data/raw/sec_edgar/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"
html = extract_primary_document(Path(p).read_text())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    elements = sp.Edgar10QParser().parse(html)

tops = [e for e in elements if type(e).__name__ == "TopSectionTitle"]
print("found", len(tops), "TopSectionTitle elements")
print()
print("public attrs:", [a for a in dir(tops[0]) if not a.startswith("_")])
print()
for e in tops:
    st = getattr(e, "section_type", None)
    ident = getattr(st, "identifier", None) if st is not None else None
    print("text=", repr(e.text[:45]), "| section_type=", repr(st), "| identifier=", repr(ident))
