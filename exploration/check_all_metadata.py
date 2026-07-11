import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chunk_filings import extract_filing_metadata, ticker_from_path

root = Path("data/raw/sec_edgar/sec-edgar-filings")
files = sorted(root.glob("*/10-K/*/full-submission.txt"))
print(f"Found {len(files)} filings\n")

rows = []
errors = []
for f in files:
    try:
        ticker = ticker_from_path(f)
        meta = extract_filing_metadata(f.read_text(), ticker)
        rows.append(meta)
    except Exception as e:
        errors.append((str(f), repr(e)))

print(f"parsed OK: {len(rows)}   errors: {len(errors)}\n")
for path, err in errors:
    print("  ERROR:", path, err)

# Tally
tickers = sorted({r["ticker"] for r in rows})
print(f"\ntickers ({len(tickers)}): {tickers}")
print("filing_types:", dict(Counter(r["filing_type"] for r in rows)))
print("fiscal_years:", dict(sorted(Counter(r["fiscal_year"] for r in rows).items())))

# Per-ticker year coverage (expect 3 each)
print("\nyears per ticker:")
by_ticker = defaultdict(list)
for r in rows:
    by_ticker[r["ticker"]].append(r["fiscal_year"])
for t in tickers:
    yrs = sorted(by_ticker[t])
    flag = "" if len(yrs) == 3 else f"  <-- {len(yrs)} filings"
    print(f"  {t:6} {yrs}{flag}")
