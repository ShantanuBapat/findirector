"""
Label the deduplicated synthetic queries with the directive labeler prompt.

For each query in data/synthetic/dedup/{code}.jsonl:
  - Call the labeling prompt (Phase 3)
  - Parse the {action_code, params, reasoning} response
  - Compare predicted code vs generator intent (the original `code`)
  - Append labeled record to data/synthetic/labeled/{code}.jsonl
  - Track disagreements separately to data/synthetic/disagreements.jsonl

Concurrency: ThreadPoolExecutor with 5 workers.
Retries: 1 retry with backoff on transient failures.

Usage:
    python -m scripts.label_synthetic_queries

Cost projection: ~$11 for ~998 unique queries.
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from anthropic import Anthropic
from dotenv import load_dotenv

from prompts.directive_labeler import (
    ACTION_CODES,
    SYSTEM_PROMPT,
    build_messages,
)


# Configuration
DEDUP_DIR = Path("data/synthetic/dedup")
LABELED_DIR = Path("data/synthetic/labeled")
DISAGREEMENTS_PATH = Path("data/synthetic/disagreements.jsonl")
NUM_WORKERS = 5
MAX_TOKENS = 512
MODEL = "claude-sonnet-4-5"


# Thread-safe lock for file writes
_write_lock = Lock()


def _strip_code_fences(text: str) -> str:
    """Strip markdown fences from labeler responses (Phase 3 defensive parsing)."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline != -1:
        text = text[first_newline + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text


def label_one_query(
    client: Anthropic, query: str, intended_code: str
) -> tuple[dict, dict]:
    """
    Label a single query. Returns (labeled_record, usage).

    On parse failure, returns a labeled_record with action_code='PARSE_ERROR'.
    On API failure, raises after one retry.
    """
    attempts = 0
    last_error = None
    while attempts < 2:
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=build_messages(query),
            )
            break
        except Exception as e:
            attempts += 1
            last_error = e
            if attempts < 2:
                # Backoff: small delay before retry
                time.sleep(2)
            else:
                raise

    response_text = response.content[0].text
    cleaned = _strip_code_fences(response_text)

    try:
        parsed = json.loads(cleaned)
        predicted_code = parsed.get("action_code", "MISSING")
        params = parsed.get("params", {})
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        predicted_code = "PARSE_ERROR"
        params = {}
        reasoning = f"Failed to parse: {response_text[:200]}"

    labeled = {
        "query": query,
        "intended_code": intended_code,
        "predicted_code": predicted_code,
        "params": params,
        "reasoning": reasoning,
        "agrees": predicted_code == intended_code,
    }

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return labeled, usage


def load_queries_for_code(code: str) -> list[dict]:
    """Load all queries for a code from the dedup file."""
    path = DEDUP_DIR / f"{code}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_labeled_record(path: Path, record: dict) -> None:
    """Thread-safe append of a labeled record to JSONL."""
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check .env.")
        return 1

    if not DEDUP_DIR.exists():
        print(f"ERROR: {DEDUP_DIR} not found. Run Phase 4 dedup first.")
        return 1

    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any existing labeled files (fresh start)
    for code in ACTION_CODES:
        out_path = LABELED_DIR / f"{code}.jsonl"
        if out_path.exists():
            print(f"  Truncating existing {out_path.name}")
            out_path.unlink()
    if DISAGREEMENTS_PATH.exists():
        DISAGREEMENTS_PATH.unlink()

    # Build full work queue: (query_text, intended_code, output_path)
    all_work: list[tuple[str, str, Path]] = []
    for code in ACTION_CODES:
        records = load_queries_for_code(code)
        out_path = LABELED_DIR / f"{code}.jsonl"
        for record in records:
            all_work.append((record["query"], code, out_path))

    print(f"Labeling {len(all_work)} queries with {NUM_WORKERS} workers.")
    print(f"Input:  {DEDUP_DIR}")
    print(f"Output: {LABELED_DIR}")
    print("=" * 80)

    client = Anthropic()

    grand_start = time.time()
    grand_input_tokens = 0
    grand_output_tokens = 0
    n_completed = 0
    n_failed = 0
    n_disagreements = 0
    code_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "agrees": 0})
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    def process(item: tuple[str, str, Path]) -> dict:
        query, intended_code, out_path = item
        try:
            labeled, usage = label_one_query(client, query, intended_code)
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "intended_code": intended_code,
                "out_path": str(out_path),
            }

        append_labeled_record(out_path, labeled)
        if not labeled["agrees"]:
            append_labeled_record(DISAGREEMENTS_PATH, labeled)

        return {
            "labeled": labeled,
            "usage": usage,
            "out_path": str(out_path),
        }

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process, item): item for item in all_work}
        for future in as_completed(futures):
            n_completed += 1
            result = future.result()

            if "error" in result:
                n_failed += 1
                print(f"  [{n_completed}/{len(all_work)}] FAILED: "
                      f"{result['error'][:100]}")
                continue

            labeled = result["labeled"]
            usage = result["usage"]
            grand_input_tokens += usage["input_tokens"]
            grand_output_tokens += usage["output_tokens"]

            intended = labeled["intended_code"]
            predicted = labeled["predicted_code"]
            code_stats[intended]["total"] += 1
            if labeled["agrees"]:
                code_stats[intended]["agrees"] += 1
            else:
                n_disagreements += 1
                confusion[(intended, predicted)] += 1

            # Compact progress every 25 queries
            if n_completed % 25 == 0 or n_completed == len(all_work):
                elapsed = time.time() - grand_start
                rate = n_completed / elapsed if elapsed > 0 else 0
                eta = (len(all_work) - n_completed) / rate if rate > 0 else 0
                print(f"  [{n_completed}/{len(all_work)}] "
                      f"elapsed={elapsed:.0f}s "
                      f"rate={rate:.1f}/s "
                      f"eta={eta:.0f}s "
                      f"disagreements={n_disagreements}")

    grand_elapsed = time.time() - grand_start
    grand_cost = (
        (grand_input_tokens / 1_000_000) * 3.0
        + (grand_output_tokens / 1_000_000) * 15.0
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nLabeled:  {n_completed - n_failed}/{len(all_work)} queries")
    print(f"Failed:   {n_failed}")
    print(f"Time:     {grand_elapsed:.1f}s ({grand_elapsed / 60:.1f} min)")
    print(f"Tokens:   input={grand_input_tokens:,}  output={grand_output_tokens:,}")
    print(f"Cost:     ${grand_cost:.4f}")
    print(f"Per query: ${grand_cost / max(n_completed - n_failed, 1):.6f}")

    print(f"\nPer-code agreement (generator intent vs labeler verdict):")
    for code in ACTION_CODES:
        s = code_stats[code]
        if s["total"] == 0:
            continue
        agree_pct = s["agrees"] / s["total"] * 100
        print(f"  {code:12s}  {s['agrees']:3d}/{s['total']:3d} = {agree_pct:5.1f}%")

    print(f"\nTotal disagreements: {n_disagreements} "
          f"({n_disagreements / max(n_completed - n_failed, 1) * 100:.1f}%)")

    if confusion:
        print(f"\nTop confusions (intended -> predicted):")
        sorted_confusions = sorted(confusion.items(), key=lambda kv: -kv[1])
        for (intended, predicted), count in sorted_confusions[:10]:
            print(f"  {intended:12s} -> {predicted:12s}  {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
