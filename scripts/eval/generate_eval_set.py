# scripts/eval/generate_eval_set.py
#
# WHAT THIS DOES:
#   Builds a DRAFT corpus-grounded eval set by generating question/answer pairs
#   FROM the project's own chunks (data/chunks/corpus.jsonl) via the Anthropic
#   API. Because each question is generated from a specific chunk, the answer is
#   guaranteed to live in the corpus AND we get the source-chunk label for free
#   (accession_number + chunk_index) — which retrieval eval needs. Output is a
#   DRAFT for human verification, not a finished set.
#
# INPUT:
#   - data/chunks/corpus.jsonl (the chunk corpus).
#   - ANTHROPIC_API_KEY in .env.
#   - CLI: --n (how many cases to try to generate), --out (output path).
#
# OUTPUT:
#   - A JSONL file of EvalCase records (default data/eval/eval_set_draft.jsonl),
#     one per line, for the human-verification pass.
#
# METHOD:
#   1. Load chunks; keep only information-dense sections and reasonably long
#      chunks (boilerplate / tiny chunks make poor questions).
#   2. Stratified sample across tickers so no company dominates.
#   3. For each sampled chunk, ask the API to produce a realistic user question,
#      the exact answer, answer_type (numeric/text), and a question_type tag —
#      returned as strict JSON.
#   4. Stamp each result with the chunk's source identity and write an EvalCase.
#
# NOTE: makes one API call per sampled chunk, so ~50 cases = ~50 calls; run time
#   is a few minutes and cost is small.

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from scripts.eval.eval_types import EvalCase, EvalSource

load_dotenv()

_MODEL = "claude-sonnet-4-5"
_MAX_TOKENS = 512

# Sections that tend to contain answerable, realistic questions.
_DENSE_SECTIONS = {"mdna", "financials", "risk_factors", "business"}
_MIN_TOKENS = 80          # skip tiny chunks — poor question material

_GEN_PROMPT = """You are helping build an evaluation set for a financial-filings \
Q&A system. Below is an excerpt from {ticker}'s fiscal {year} SEC 10-K \
({section} section).

Write ONE realistic question a user might ask that this excerpt answers, along \
with the exact answer grounded in the excerpt.

Rules:
- The question must be answerable SOLELY from this excerpt.
- Phrase it as a natural user question (do not reference "the excerpt").
- Prefer specific, checkable facts or figures over vague questions.
- Give the answer concisely and exactly as supported by the text.

Return STRICT JSON only, no markdown, with these keys:
{{"question": "...", "answer": "...", "answer_type": "numeric" or "text", \
"question_type": "lookup_fact" or "definition"}}

Excerpt:
\"\"\"
{text}
\"\"\""""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s


def _load_dense_chunks(corpus_path: Path) -> list[dict]:
    """Load chunks, keeping only dense sections and reasonably long ones."""
    chunks = []
    for line in corpus_path.open():
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("section") in _DENSE_SECTIONS and c.get("n_tokens", 0) >= _MIN_TOKENS:
            chunks.append(c)
    return chunks


def _stratified_sample(chunks: list[dict], n: int, seed: int = 13) -> list[dict]:
    """Sample ~n chunks spread across tickers so none dominates."""
    rng = random.Random(seed)
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_ticker[c["ticker"]].append(c)
    tickers = list(by_ticker)
    rng.shuffle(tickers)

    # Ceiling division so we collect at least n across all tickers; the final
    # sample[:n] then trims to exactly n. (Floor division would undershoot —
    # e.g. 50 // 20 = 2 per ticker × 20 = only 40.)
    per_ticker = max(1, -(-n // len(tickers)))   # -(-a // b) = ceil(a / b)
    sample: list[dict] = []
    for t in tickers:
        pool = by_ticker[t]
        rng.shuffle(pool)
        sample.extend(pool[:per_ticker])
    rng.shuffle(sample)
    return sample[:n]


def _generate_one(client, chunk: dict) -> dict | None:
    """Call the API to produce a Q/A for one chunk; return parsed dict or None."""
    prompt = _GEN_PROMPT.format(
        ticker=chunk["ticker"], year=chunk["fiscal_year"],
        section=chunk.get("section"), text=chunk["text"][:4000],
    )
    resp = client.messages.create(
        model=_MODEL, max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(_strip_fences(resp.content[0].text))
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--corpus", default="data/chunks/corpus.jsonl")
    ap.add_argument("--out", default="data/eval/eval_set_draft.jsonl")
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chunks = _load_dense_chunks(Path(args.corpus))
    print(f"{len(chunks)} dense chunks available; sampling {args.n}")
    sample = _stratified_sample(chunks, args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w") as out:
        for i, chunk in enumerate(sample, 1):
            gen = _generate_one(client, chunk)
            if gen is None:
                print(f"  [{i}/{len(sample)}] skipped (bad JSON)")
                continue
            case = EvalCase(
                id=f"eval_{i:03d}",
                question=gen.get("question", ""),
                expected_answer=gen.get("answer", ""),
                answer_type=gen.get("answer_type", "text"),
                source=EvalSource(
                    ticker=chunk["ticker"], fiscal_year=chunk["fiscal_year"],
                    section=chunk.get("section"),
                    chunk_id=f'{chunk["accession_number"]}#{chunk["chunk_index"]}',
                ),
                directive_params={
                    "company": chunk["ticker"], "year": chunk["fiscal_year"],
                },
                question_type=gen.get("question_type", "lookup_fact"),
            )
            out.write(json.dumps(asdict(case)) + "\n")
            written += 1
            print(f"  [{i}/{len(sample)}] {chunk['ticker']} {chunk['fiscal_year']}"
                  f" {chunk.get('section')}: {case.question[:60]}")

    print(f"\nDone: {written} draft eval cases -> {out_path}")
    print("NEXT: human-verify — cull trivial/broken items, confirm answers.")


if __name__ == "__main__":
    main()