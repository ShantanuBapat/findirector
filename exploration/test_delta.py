import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chunk_filings import chunk_corpus, load_processed_accessions

ROOT = "data/raw/sec_edgar/sec-edgar-filings"
CORPUS = "data/chunks/corpus.jsonl"

# Clean slate for the test
p = Path(CORPUS)
if p.exists():
    p.unlink()

print("=== RUN 1: empty corpus, should process all 60 ===")
t0 = time.time()
processed, skipped, errors = chunk_corpus(ROOT, CORPUS, verbose=True)
print(f"-> processed={processed} skipped={skipped} errors={len(errors)}")
print(f"corpus lines: {sum(1 for _ in open(CORPUS))}")
print(f"accessions in corpus: {len(load_processed_accessions(CORPUS))}")
print(f"run 1 wall time: {time.time()-t0:.0f}s\n")

print("=== RUN 2: full corpus exists, should process 0, skip 60 ===")
t0 = time.time()
processed, skipped, errors = chunk_corpus(ROOT, CORPUS, verbose=True)
print(f"-> processed={processed} skipped={skipped} errors={len(errors)}")
print(f"corpus lines (unchanged?): {sum(1 for _ in open(CORPUS))}")
print(f"run 2 wall time: {time.time()-t0:.1f}s  <- should be ~instant\n")

print("=== RUN 3: force=True, should reprocess all 60 ===")
processed, skipped, errors = chunk_corpus(ROOT, CORPUS, force=True, verbose=True)
print(f"-> processed={processed} skipped={skipped} errors={len(errors)}")
print(f"corpus lines (rebuilt, no dupes?): {sum(1 for _ in open(CORPUS))}")
