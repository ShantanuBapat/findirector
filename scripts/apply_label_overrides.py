"""
Apply manual label overrides for known labeler blind spots.

The Phase 3 labeler doesn't know our corpus boundaries (20 specific
tickers, 10-K only). For queries that ARE out-of-scope due to corpus
constraints, the labeler classifies them as `lookup` when they should
be `decline` (out_of_scope).

This script identifies those queries by query text match (not query
index, which is unstable across runs) and overrides the `predicted_code`
to `decline` with the correct `params.reason`.

Currently overrides 3 queries identified via Phase 5 disagreement review:
- 2 Netflix queries (company not in corpus)
- 1 Pfizer 10-Q query (wrong filing type in corpus)

Usage:
    python -m scripts.apply_label_overrides
"""

import json
import sys
from pathlib import Path

from prompts.directive_labeler import ACTION_CODES


LABELED_DIR = Path("data/synthetic/labeled")
DISAGREEMENTS_PATH = Path("data/synthetic/disagreements.jsonl")


# Queries to override: {normalized_query: (new_code, new_params, justification)}
# We match on lowercase-stripped query text to be robust to whitespace.
OVERRIDES: dict[str, tuple[str, dict, str]] = {
    "tell me about netflix's streaming subscriber growth in 2023": (
        "decline",
        {"reason": "out_of_scope"},
        "Netflix not in our 20-ticker corpus",
    ),
    "what was netflix's revenue in fiscal 2023?": (
        "decline",
        {"reason": "out_of_scope"},
        "Netflix not in our 20-ticker corpus",
    ),
    "what does pfizer's 10-q say about their pipeline?": (
        "decline",
        {"reason": "out_of_scope"},
        "10-Q filings not in our corpus (10-K only)",
    ),
}


def normalize(s: str) -> str:
    """Normalize for query matching."""
    return " ".join(s.lower().split())


def main() -> int:
    if not LABELED_DIR.exists():
        print(f"ERROR: {LABELED_DIR} not found. Run Phase 5 labeling first.")
        return 1

    n_overridden = 0
    n_already_correct = 0
    n_not_found = 0

    # Process each labeled file
    for code in ACTION_CODES:
        path = LABELED_DIR / f"{code}.jsonl"
        if not path.exists():
            continue

        # Read all records
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Apply overrides where query matches
        modified = False
        for record in records:
            normalized = normalize(record["query"])
            if normalized in OVERRIDES:
                new_code, new_params, justification = OVERRIDES[normalized]
                old_code = record["predicted_code"]

                if old_code == new_code:
                    n_already_correct += 1
                    continue

                # Apply override
                record["predicted_code"] = new_code
                record["params"] = new_params
                record["reasoning"] = (
                    f"[OVERRIDE] {justification}. Original labeler said: "
                    f"{record['reasoning'][:150]}"
                )
                record["agrees"] = (record["intended_code"] == new_code)
                record["override_applied"] = True

                modified = True
                n_overridden += 1
                print(f"  Overrode in {path.name}: "
                      f"{record['query'][:60]!r} {old_code} -> {new_code}")

        # Write back if modified
        if modified:
            with open(path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

    # Check for overrides that didn't find a target
    found_queries = set()
    for code in ACTION_CODES:
        path = LABELED_DIR / f"{code}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    found_queries.add(normalize(record["query"]))

    for override_query in OVERRIDES:
        if override_query not in found_queries:
            n_not_found += 1
            print(f"  WARNING: override target not found: {override_query!r}")

    # Regenerate disagreements.jsonl from current labeled files
    print(f"\nRegenerating disagreements.jsonl...")
    disagreements = []
    for code in ACTION_CODES:
        path = LABELED_DIR / f"{code}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if not record.get("agrees", False):
                        disagreements.append(record)

    with open(DISAGREEMENTS_PATH, "w") as f:
        for r in disagreements:
            f.write(json.dumps(r) + "\n")

    print(f"\nSummary:")
    print(f"  Overrides applied:      {n_overridden}")
    print(f"  Already correct:        {n_already_correct}")
    print(f"  Override targets missing: {n_not_found}")
    print(f"  Disagreements remaining: {len(disagreements)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
