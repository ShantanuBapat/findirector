import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chunk_filings import accession_from_path

root = Path("data/raw/sec_edgar/sec-edgar-filings")
files = sorted(root.glob("*/10-K/*/full-submission.txt"))
print(f"{len(files)} filings\n")

# Accession format: NNNNNNNNNN-NN-NNNNNN
fmt = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_HEADER_ACC = re.compile(r"^ACCESSION NUMBER:\s*(\S+)", re.MULTILINE)

path_accs = []
mismatches = []
malformed = []

for f in files:
    acc = accession_from_path(f)
    path_accs.append(acc)
    if not fmt.match(acc):
        malformed.append((str(f), acc))
    # Cross-check against the header's ACCESSION NUMBER
    m = _HEADER_ACC.search(f.read_text()[:2000])
    header_acc = m.group(1) if m else None
    if header_acc != acc:
        mismatches.append((str(f), acc, header_acc))

dupes = [a for a, c in Counter(path_accs).items() if c > 1]

print(f"unique accessions: {len(set(path_accs))} / {len(path_accs)}")
print(f"duplicates: {dupes if dupes else 'none'}")
print(f"malformed: {len(malformed)}")
for f, a in malformed:
    print("   ", a, f)
print(f"path vs header mismatches: {len(mismatches)}")
for f, pa, ha in mismatches[:10]:
    print(f"    path={pa} header={ha}  {f}")
