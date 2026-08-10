# scripts/eval/run_generation_eval.py
#
# WHAT THIS DOES:
#   Evaluates the END-TO-END answer quality of FinDirector against a
#   corpus-grounded eval set. For each case it runs the REAL pipeline
#   (Orchestrator.answer) to get the system's answer, then scores it against the
#   known-correct answer using a method matched to answer_type:
#     - numeric -> normalized near-match (strip $/commas, scale million/billion,
#                  compare within tolerance). No LLM needed.
#     - text    -> LLM-as-judge (claude-sonnet-4-5): asks whether the system's
#                  answer is consistent with the expected answer. Handles
#                  paraphrase, which exact string match cannot.
#   Reports correctness overall and split by answer_type. Case-set-agnostic.
#
# INPUT:
#   --eval-set (JSONL of EvalCase records, default data/eval/eval_set.jsonl)
#   --n        (optional: only run the first n cases — for a cheap test run)
#   Requires a running pgvector DB and ANTHROPIC_API_KEY (router, generation, and
#   the judge all call the API).
#
# OUTPUT:
#   - A scorecard printed to the terminal (overall %, numeric %, text %).
#   - A timestamped results/ markdown file with per-case verdicts, for tracking.
#
# COST NOTE: each case = full pipeline (router + generation API calls); each text
#   case additionally = one judge call. ~40 cases ≈ a few minutes, small cost.
#   Use --n to test on a small subset first.

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

from scripts.embed.local_embedder import LocalEmbedder
from scripts.generation.anthropic_generator import AnthropicGenerator
from scripts.orchestration.orchestrator import Orchestrator
from scripts.routing.anthropic_router import AnthropicRouter
from scripts.store.pgvector_store import PgVectorStore

_JUDGE_MODEL = "claude-sonnet-4-5"
_NUMERIC_TOLERANCE = 0.01   # 1% relative tolerance for numeric near-match

# Words that scale a number, mapped to their multiplier.
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def _extract_numbers(text: str) -> list[float]:
    """Extract ALL numeric values from text, applying an adjacent scale word.

    Returns every number found (in base units). A scale word (million/billion/…)
    is applied only when it immediately follows the number (within a few chars),
    so '10-K' does not get scaled by a 'million' appearing later in the sentence.
    """
    if not text:
        return []
    values = []
    for m in re.finditer(r"(\d[\d,]*\.?\d*)", text):
        raw = m.group(1).replace(",", "")
        # Skip things that look like a filing form (e.g. '10-K', '10-Q').
        if re.match(r"10-[KQ]", text[m.start():m.start() + 4]):
            continue
        value = float(raw)
        # Only apply a scale word if it appears immediately after the number.
        tail = text[m.end():m.end() + 12].lower()
        for word, mult in _SCALES.items():
            if re.match(rf"\s*{word}", tail):
                value *= mult
                break
        values.append(value)
    return values


def _numeric_match(expected: str, actual: str) -> bool:
    """True if the expected number appears among actual's numbers (within tol)."""
    exp_nums = _extract_numbers(expected)
    if not exp_nums:
        return False
    target = exp_nums[0]                       # the expected value
    for a in _extract_numbers(actual):
        if target == 0:
            if abs(a) < 1e-9:
                return True
        elif abs(a - target) / abs(target) <= _NUMERIC_TOLERANCE:
            return True
    return False


def _judge_text(client, question: str, expected: str, actual: str) -> bool:
    """LLM-as-judge: is `actual` consistent with `expected` for this question?"""
    prompt = (
        "You are grading a financial-QA system's answer.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"System's answer: {actual}\n\n"
        "Is the system's answer correct and consistent with the expected answer? "
        "Minor wording or formatting differences are fine; judge the substance. "
        "Reply with exactly CORRECT or INCORRECT, then a brief reason."
    )
    resp = client.messages.create(
        model=_JUDGE_MODEL, max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = resp.content[0].text.strip().upper()
    return verdict.startswith("CORRECT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval/eval_set.jsonl")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.eval_set).open() if l.strip()]
    if args.n:
        cases = cases[:args.n]
    print(f"loaded {len(cases)} eval cases from {args.eval_set}")

    orch = Orchestrator(
        router=AnthropicRouter(), embedder=LocalEmbedder(),
        store=PgVectorStore(), generator=AnthropicGenerator(),
    )
    import anthropic
    import os
    judge_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    start = time.time()
    per_case = []
    for i, case in enumerate(cases, 1):
        answer = orch.answer(case["question"])
        actual = answer.text
        expected = case["expected_answer"]
        atype = case["answer_type"]

        if atype == "numeric":
            correct = _numeric_match(expected, actual)
        else:
            correct = _judge_text(judge_client, case["question"], expected, actual)

        per_case.append({
            "id": case["id"], "ticker": case["source"]["ticker"],
            "answer_type": atype, "status": answer.status, "correct": correct,
            "expected": expected, "actual": actual[:120],
        })
        print(f"  [{i}/{len(cases)}] {case['id']} ({atype}): "
              f"{'✓' if correct else '✗'}")

    elapsed = time.time() - start
    n = len(cases)
    n_correct = sum(1 for c in per_case if c["correct"])
    num_cases = [c for c in per_case if c["answer_type"] == "numeric"]
    txt_cases = [c for c in per_case if c["answer_type"] != "numeric"]
    num_acc = (sum(c["correct"] for c in num_cases) / len(num_cases)
               if num_cases else 0.0)
    txt_acc = (sum(c["correct"] for c in txt_cases) / len(txt_cases)
               if txt_cases else 0.0)

    print(f"\n{'='*56}")
    print(f"GENERATION EVAL  (n={n})")
    print(f"  overall correctness: {n_correct/n:.1%}  ({n_correct}/{n})")
    print(f"  numeric: {num_acc:.1%} ({len(num_cases)} cases)")
    print(f"  text:    {txt_acc:.1%} ({len(txt_cases)} cases)")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"{'='*56}")
    wrong = [c for c in per_case if not c["correct"]]
    if wrong:
        print(f"\nINCORRECT ({len(wrong)}):")
        for c in wrong:
            print(f"  {c['id']} ({c['ticker']}, {c['answer_type']}, "
                  f"status={c['status']})")
            print(f"    expected: {c['expected'][:80]}")
            print(f"    actual:   {c['actual']}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"generation_eval_{date.today().isoformat()}.md"
    with out.open("w") as f:
        f.write(f"# Generation Eval — {date.today().isoformat()}\n\n")
        f.write(f"- eval set: `{args.eval_set}` ({n} cases)\n")
        f.write(f"- **overall: {n_correct/n:.1%}** ({n_correct}/{n})\n")
        f.write(f"- numeric: {num_acc:.1%} ({len(num_cases)}); "
                f"text: {txt_acc:.1%} ({len(txt_cases)})\n")
        f.write(f"- judge: {_JUDGE_MODEL}; elapsed {elapsed:.1f}s\n\n")
        f.write("| id | ticker | type | correct | expected | actual |\n")
        f.write("|----|--------|------|---------|----------|--------|\n")
        for c in per_case:
            f.write(f"| {c['id']} | {c['ticker']} | {c['answer_type']} | "
                    f"{'✓' if c['correct'] else '✗'} | "
                    f"{c['expected'][:40]} | {c['actual'][:40]} |\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
