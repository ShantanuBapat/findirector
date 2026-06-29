"""
Test the directive-labeling prompt on a curated set of edge-case queries.

Each query has an expected action_code. The script:
  1. Sends each query to Claude using the labeling prompt
  2. Parses the JSON response
  3. Compares predicted code vs expected code
  4. Prints per-query results + summary stats + cost

Usage:
    python scripts/test_directive_labeler.py
"""

import json
import os
import sys
import time

from anthropic import Anthropic
from dotenv import load_dotenv

from prompts.directive_labeler import SYSTEM_PROMPT, build_messages


# -----------------------------------------------------------------------------
# Test cases — curated from our earlier edge-case calibration sessions
# -----------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str]] = [
    # (query, expected_action_code)

    # smalltalk — easy
    ("Hi there!", "smalltalk"),
    ("Thanks, that was really helpful.", "smalltalk"),

    # meta — questions about the system
    ("What can you do?", "meta"),
    ("What's the difference between a 10-K and a 10-Q?", "meta"),

    # lookup — single fact, single company
    ("What was Apple's R&D spending in fiscal 2023?", "lookup"),
    ("What does Microsoft mean by 'commercial cloud' in their filings?", "lookup"),

    # compute — single company arithmetic
    ("What's Apple's R&D as a percentage of revenue for 2023?", "compute"),

    # research — multi-company OR multi-period
    ("Compare Tesla's debt levels from 2021 to 2024.", "research"),
    ("How does Microsoft's gross margin compare to Apple's in 2024?", "research"),

    # clarify — ambiguous query
    ("How did the company do last quarter?", "clarify"),

    # decline — various reasons
    ("Should I buy AAPL?", "decline"),
    ("Will Tesla's stock go up next year?", "decline"),
    ("What's the weather today?", "decline"),
]


# -----------------------------------------------------------------------------
# Per-query labeling
# -----------------------------------------------------------------------------

def label_query(client: Anthropic, query: str) -> tuple[dict, dict]:
    """
    Call Claude with the labeling prompt for one query.

    Returns:
        (parsed_response_dict, usage_dict)
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=build_messages(query),
    )

    response_text = response.content[0].text.strip()

    # Strip markdown code fences if present (Claude sometimes adds them
    # despite the prompt instructions). Handles ```json...``` and ```...```.
    if response_text.startswith('```'):
        # Remove leading fence (with or without language tag)
        response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text
        # Remove trailing fence
        if response_text.endswith('```'):
            response_text = response_text[:-3].strip()

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        parsed = {
            "action_code": "PARSE_ERROR",
            "params": {},
            "reasoning": f"Failed to parse JSON: {e}. Raw response: {response_text[:200]}",
        }

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return parsed, usage


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check .env.")
        return 1

    client = Anthropic()

    print(f"Testing directive labeler on {len(TEST_CASES)} curated queries.\n")
    print("=" * 80)

    correct = 0
    total_input_tokens = 0
    total_output_tokens = 0
    failures: list[tuple[int, str, str, str, str]] = []

    start_time = time.time()

    for i, (query, expected) in enumerate(TEST_CASES, start=1):
        parsed, usage = label_query(client, query)
        predicted = parsed.get("action_code", "MISSING")
        reasoning = parsed.get("reasoning", "(no reasoning)")

        status = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        else:
            failures.append((i, query, expected, predicted, reasoning))

        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]

        print(f"[{i:2d}] {status}  expected={expected:<10}  predicted={predicted:<10}")
        print(f"     query: {query}")
        if predicted != expected:
            print(f"     reasoning: {reasoning}")
        print()

    elapsed = time.time() - start_time
    accuracy = correct / len(TEST_CASES) * 100

    print("=" * 80)
    print(f"Accuracy:  {correct}/{len(TEST_CASES)} = {accuracy:.1f}%")
    print(f"Elapsed:   {elapsed:.1f}s ({elapsed / len(TEST_CASES):.1f}s per query)")
    print(f"Tokens:    input={total_input_tokens:,}  output={total_output_tokens:,}")

    # Cost estimate (Sonnet 4.5: $3/M input, $15/M output)
    cost = (total_input_tokens / 1_000_000) * 3.0 + (total_output_tokens / 1_000_000) * 15.0
    print(f"Cost:      ${cost:.4f}")
    print()

    if failures:
        print("FAILURES:")
        for i, query, expected, predicted, reasoning in failures:
            print(f"  [{i}] expected={expected}, predicted={predicted}")
            print(f"      query: {query}")
            print(f"      reasoning: {reasoning}")
            print()

    return 0 if accuracy >= 90 else 1


if __name__ == "__main__":
    sys.exit(main())
