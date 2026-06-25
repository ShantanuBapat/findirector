"""
Download SEC 10-K filings for the 20 S&P 500 companies in the FinDirector corpus.

Usage:
    python scripts/download_filings.py

Downloads to ./data/raw/sec_edgar/<TICKER>/10-K/

Note: SEC EDGAR requires identifying contact info in the User-Agent header.
This is enforced by SEC; non-identifying requests are rate-limited or blocked.
"""

import time
from pathlib import Path
from sec_edgar_downloader import Downloader

# ---- Configuration ----

# 20 S&P 500 companies across diverse sectors
TICKERS = [
    # Big Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Finance
    "JPM", "BAC", "BRK-B", "V",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "KO", "PG",
    # Energy
    "XOM", "CVX",
    # Industrial / Auto
    "TSLA", "BA",
]

# How many years of 10-Ks per company
NUM_FILINGS = 3

# Local download directory (gitignored)
DOWNLOAD_DIR = Path("data/raw/sec_edgar")

# Identifying info for SEC EDGAR (required by SEC)
# Update these to your real name/email before running.
COMPANY_NAME = "FinDirector Personal Learning Project"
CONTACT_EMAIL = "shantanu_bapat@hotmail.com"


def main() -> None:
    """Download recent 10-K filings for all configured tickers."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(COMPANY_NAME, CONTACT_EMAIL, str(DOWNLOAD_DIR))

    start_time = time.time()
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, ticker in enumerate(TICKERS, start=1):
        print(f"[{i}/{len(TICKERS)}] Downloading 10-Ks for {ticker}...")
        try:
            num_downloaded = dl.get("10-K", ticker, limit=NUM_FILINGS)
            print(f"  ✓ {num_downloaded} filing(s) downloaded for {ticker}")
            successes.append(ticker)
        except Exception as e:
            print(f"  ✗ Failed for {ticker}: {e}")
            failures.append((ticker, str(e)))

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Successes ({len(successes)}): {', '.join(successes)}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for ticker, error in failures:
            print(f"  {ticker}: {error}")


if __name__ == "__main__":
    main()
