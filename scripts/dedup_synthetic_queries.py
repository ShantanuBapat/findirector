"""
Deduplicate synthetic queries within each code.

Reads data/synthetic/raw/{code}.jsonl files and writes deduplicated
versions to data/synthetic/dedup/{code}.jsonl. Uses normalized
(lowercase + whitespace-stripped) query text for comparison so trivial
variations are merged, but preserves the ORIGINAL casing/spacing in
the output.

Cross-code duplicates are NOT removed (a smalltalk "Hi" and a clarify
"Hi" could plausibly coexist if they were generated for different codes).

Usage:
    python -m scripts.dedup_synthetic_queries
"""

import json
import sys
from pathlib import Path

from prompts.directive_labeler import ACTION_CODES


RAW_DIR = Path("data/synthetic/raw")
DEDUP_DIR = Path("data/synthetic/dedup")


def normalize(query: str) -> str:
    """Normalize for duplicate detection (case + whitespace insensitive)."""
    return " ".join(query.lower().split())


def dedup_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """
    Deduplicate one JSONL file.

    Returns (n_total_read, n_unique_written).
    """
    seen_normalized: set[str] = set()
    unique_records: list[dict] = []
    n_total = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_total += 1
            normalized = normalize(record["query"])
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_records.append(record)

    with open(output_path, "w") as f:
        for record in unique_records:
            f.write(json.dumps(record) + "\n")

    return n_total, len(unique_records)


def main() -> int:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist. Run generation first.")
        return 1

    DEDUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Deduplicating queries from {RAW_DIR} -> {DEDUP_DIR}")
    print("=" * 80)

    grand_total_in = 0
    grand_total_out = 0
    per_code: dict[str, tuple[int, int]] = {}

    for code in ACTION_CODES:
        input_path = RAW_DIR / f"{code}.jsonl"
        output_path = DEDUP_DIR / f"{code}.jsonl"

        if not input_path.exists():
            print(f"  WARNING: {input_path} missing — skipping")
            continue

        n_in, n_out = dedup_file(input_path, output_path)
        n_dupes = n_in - n_out
        dup_pct = (n_dupes / n_in * 100) if n_in else 0
        per_code[code] = (n_in, n_out)
        grand_total_in += n_in
        grand_total_out += n_out

        print(f"  {code:12s}  {n_in:4d} in  ->  {n_out:4d} unique  "
              f"({n_dupes} dupes removed, {dup_pct:.1f}%)")

    print("=" * 80)
    grand_dupes = grand_total_in - grand_total_out
    grand_pct = (grand_dupes / grand_total_in * 100) if grand_total_in else 0
    print(f"\nTotal: {grand_total_in} read, {grand_total_out} unique kept "
          f"({grand_dupes} dupes removed, {grand_pct:.1f}%)")

    # Estimated labeling cost savings
    saved_calls = grand_dupes
    saved_cost = saved_calls * 0.011  # ~$0.011 per labeling call (Phase 3 measured)
    print(f"Estimated Phase 5 cost savings: ${saved_cost:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
