"""
Generate synthetic queries for all 7 action codes at scale.

For each code:
  - Calls the generator in batches of 10 queries
  - Appends each batch to data/synthetic/raw/{code}.jsonl
  - Continues until target count is reached
  - Prints progress + per-code stats at the end

Each line in the output JSONL is a JSON object:
    {"code": "lookup", "query": "What was Apple's R&D in 2023?", "batch_id": 3}

Usage:
    python -m scripts.generate_synthetic_queries

Cost projection: ~$1.50 total for 200 queries per code.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from prompts.directive_labeler import ACTION_CODES
from prompts.synthetic_query_generator import generate_queries


# Configuration
TARGET_PER_CODE: int = 200
QUERIES_PER_CALL: int = 10
OUTPUT_DIR = Path("data/synthetic/raw")


def generate_for_code(
    client: Anthropic,
    code: str,
    target: int,
    queries_per_call: int,
    output_file: Path,
) -> tuple[int, int, int]:
    """
    Generate `target` queries for a single code, saving each batch to file.

    Returns (num_queries_written, total_input_tokens, total_output_tokens).
    """
    n_written = 0
    n_calls = 0
    total_input = 0
    total_output = 0
    batch_id = 0

    # Open in append mode so we add to existing file if re-run
    with open(output_file, "a") as f:
        while n_written < target:
            batch_id += 1
            n_calls += 1
            remaining = target - n_written
            n_this_batch = min(queries_per_call, remaining)

            try:
                queries, usage = generate_queries(
                    client, code, n_this_batch
                )
            except Exception as e:
                print(f"      WARNING: batch {batch_id} failed: {e}")
                # Sleep briefly and retry once
                time.sleep(2)
                try:
                    queries, usage = generate_queries(
                        client, code, n_this_batch
                    )
                except Exception as e2:
                    print(f"      ERROR: retry also failed: {e2}. Skipping.")
                    continue

            # Append each query as a JSONL line
            for q in queries:
                record = {
                    "code": code,
                    "query": q.strip(),
                    "batch_id": batch_id,
                }
                f.write(json.dumps(record) + "\n")
            f.flush()  # Force write to disk; durability against crashes

            n_written += len(queries)
            total_input += usage["input_tokens"]
            total_output += usage["output_tokens"]

            print(f"      batch {batch_id}: {len(queries)} queries "
                  f"(total {n_written}/{target}) — "
                  f"in={usage['input_tokens']}, out={usage['output_tokens']}")

    return n_written, total_input, total_output


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check .env.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Anthropic()

    print(f"Generating {TARGET_PER_CODE} queries per code "
          f"({TARGET_PER_CODE * len(ACTION_CODES)} total).")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Queries per API call: {QUERIES_PER_CALL}")
    print("=" * 80)

    grand_start = time.time()
    grand_total_input = 0
    grand_total_output = 0
    per_code_counts: dict[str, int] = {}

    for code in ACTION_CODES:
        print(f"\n--- {code.upper()} ---")
        output_file = OUTPUT_DIR / f"{code}.jsonl"

        # If file exists with content, start fresh (don't append to partial)
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  WARNING: {output_file} exists with content; truncating.")
            output_file.unlink()

        code_start = time.time()
        n_written, in_tokens, out_tokens = generate_for_code(
            client, code, TARGET_PER_CODE, QUERIES_PER_CALL, output_file
        )
        code_elapsed = time.time() - code_start
        code_cost = (in_tokens / 1_000_000) * 3.0 + (out_tokens / 1_000_000) * 15.0

        per_code_counts[code] = n_written
        grand_total_input += in_tokens
        grand_total_output += out_tokens

        print(f"  Done: {n_written} queries in {code_elapsed:.1f}s "
              f"(${code_cost:.4f})")

    grand_elapsed = time.time() - grand_start
    grand_cost = (grand_total_input / 1_000_000) * 3.0 + \
                 (grand_total_output / 1_000_000) * 15.0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPer-code counts:")
    for code in ACTION_CODES:
        n = per_code_counts.get(code, 0)
        print(f"  {code:12s}  {n} queries")

    total_queries = sum(per_code_counts.values())
    print(f"\nTotal queries: {total_queries}")
    print(f"Wall time:     {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"Tokens:        input={grand_total_input:,}  output={grand_total_output:,}")
    print(f"Cost:          ${grand_cost:.4f}")
    print(f"Per query:     ${grand_cost/total_queries:.6f}")

    # Quick dedup check per code
    print(f"\nDeduplication check (within each code):")
    for code in ACTION_CODES:
        path = OUTPUT_DIR / f"{code}.jsonl"
        if not path.exists():
            continue
        queries_in_code = []
        with open(path) as f:
            for line in f:
                queries_in_code.append(json.loads(line)["query"].lower().strip())
        n_total = len(queries_in_code)
        n_unique = len(set(queries_in_code))
        dup_rate = (n_total - n_unique) / n_total * 100 if n_total else 0
        print(f"  {code:12s}  {n_total} total, {n_unique} unique "
              f"({dup_rate:.1f}% duplicates)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
