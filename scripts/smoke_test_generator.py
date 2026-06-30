"""
Smoke test the synthetic query generator.

Generates 5 queries per action code (35 total) so we can inspect quality
before scaling to ~200 per code. Reads .env for ANTHROPIC_API_KEY.

Usage:
    python -m scripts.smoke_test_generator
"""

import os
import sys
import time
from collections import defaultdict

from anthropic import Anthropic
from dotenv import load_dotenv

from prompts.directive_labeler import ACTION_CODES
from prompts.synthetic_query_generator import generate_queries


# How many queries per code for the smoke test
QUERIES_PER_CODE: int = 5


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check .env.")
        return 1

    client = Anthropic()

    print(f"Smoke test: generating {QUERIES_PER_CODE} queries per code "
          f"({QUERIES_PER_CODE * len(ACTION_CODES)} total).\n")
    print("=" * 80)

    all_results: dict[str, list[str]] = defaultdict(list)
    total_input_tokens = 0
    total_output_tokens = 0
    failures: list[tuple[str, str]] = []

    start_time = time.time()

    for code in ACTION_CODES:
        print(f"\n--- {code.upper()} ---")
        try:
            queries, usage = generate_queries(client, code, QUERIES_PER_CODE)
            all_results[code] = queries
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]

            for i, q in enumerate(queries, start=1):
                print(f"  [{i}] {q}")
        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append((code, str(e)))

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"Elapsed:   {elapsed:.1f}s ({elapsed / len(ACTION_CODES):.1f}s per code)")
    print(f"Tokens:    input={total_input_tokens:,}  output={total_output_tokens:,}")

    # Cost estimate (Sonnet 4.5: $3/M input, $15/M output)
    cost = (total_input_tokens / 1_000_000) * 3.0 + (total_output_tokens / 1_000_000) * 15.0
    print(f"Cost:      ${cost:.4f}")

    if failures:
        print(f"\nFAILURES: {len(failures)}")
        for code, err in failures:
            print(f"  {code}: {err[:200]}")
        return 1

    # Sanity-check: did each code produce exactly QUERIES_PER_CODE queries?
    for code in ACTION_CODES:
        n_got = len(all_results.get(code, []))
        if n_got != QUERIES_PER_CODE:
            print(f"WARNING: {code} produced {n_got} queries, expected {QUERIES_PER_CODE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
