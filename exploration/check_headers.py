import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

root = Path("data/raw/sec_edgar/sec-edgar-filings")

# Sample a few different companies (varied fiscal year-ends).
tickers = ["AAPL", "MSFT", "JPM", "WMT", "XOM"]
for ticker in tickers:
    # Grab one filing for this ticker
    files = sorted((root / ticker / "10-K").glob("*/full-submission.txt"))
    if not files:
        print(f"{ticker}: NO FILINGS FOUND")
        continue
    # Read just the header region (first ~4000 chars is plenty)
    head = files[0].read_text()[:4000]
    print(f"=== {ticker} ({files[0].parent.name}) ===")
    for line in head.splitlines():
        s = line.strip()
        if s.startswith("CONFORMED SUBMISSION TYPE") or s.startswith("CONFORMED PERIOD OF REPORT"):
            print("  ", s)
    print()
