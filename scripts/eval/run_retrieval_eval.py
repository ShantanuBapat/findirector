# scripts/eval/run_retrieval_eval.py
#
# WHAT THIS DOES:
#   Evaluates the RETRIEVAL stage of FinDirector against a corpus-grounded eval
#   set. For each case it runs the REAL retrieval path (the same retrieve() the
#   orchestrator uses), then checks whether the case's labeled source chunk
#   (accession_number#chunk_index) appears in the top-k results, and at what rank.
#   Reports hit-rate and MRR. Case-set-agnostic: it runs whatever JSONL you point
#   it at, so the same harness can later score an adversarial set (Track F).
#
# INPUT:
#   --eval-set (JSONL of EvalCase records, default data/eval/eval_set.jsonl)
#   --k        (top-k to retrieve/score, default 5)
#   Requires a running pgvector DB (docker compose up) and ANTHROPIC_API_KEY in
#   .env (the embedder is local; no router/generation calls here).
#
# OUTPUT:
#   - A scorecard printed to the terminal (hit-rate, MRR, per-case ranks).
#   - A timestamped results file under results/ (markdown), for tracking over
#     time — consistent with the directive-model eval reports.
#
# METRICS (plain English):
#   hit@k : fraction of cases where the labeled chunk was retrieved in the top-k
#           ("did we fetch the answer's home?").
#   MRR   : mean of 1/rank of the labeled chunk (rank 1 -> 1.0, rank 3 -> 0.33,
#           not found -> 0). Rewards ranking the right chunk HIGHER, not just
#           present.

import argparse
import json
import time
from datetime import date
from pathlib import Path

from scripts.embed.local_embedder import LocalEmbedder
from scripts.retrieval.retrieve import retrieve
from scripts.store.pgvector_store import PgVectorStore


def _chunk_id(row: dict) -> str:
    """Build the composite chunk identity used as the retrieval label."""
    return f'{row["accession_number"]}#{row["chunk_index"]}'


def _rank_of_labeled_chunk(results: list[dict], labeled_id: str) -> int | None:
    """Return the 1-based rank of the labeled chunk in results, or None."""
    for rank, row in enumerate(results, start=1):
        if _chunk_id(row) == labeled_id:
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval/eval_set.jsonl")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.eval_set).open() if l.strip()]
    print(f"loaded {len(cases)} eval cases from {args.eval_set}")

    embedder = LocalEmbedder()
    store = PgVectorStore()

    start = time.time()
    per_case = []
    hits = 0
    reciprocal_ranks = []

    for case in cases:
        labeled_id = case["source"]["chunk_id"]
        params = case.get("directive_params", {})
        result = retrieve(case["question"], params, embedder, store, k=args.k)

        # retrieve() returns {"status": "ok"/"decline", "chunks": [...]}
        results = result.get("chunks", []) if result.get("status") == "ok" else []
        rank = _rank_of_labeled_chunk(results, labeled_id)

        hit = rank is not None
        hits += 1 if hit else 0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        per_case.append({
            "id": case["id"], "ticker": case["source"]["ticker"],
            "rank": rank, "hit": hit,
        })

    elapsed = time.time() - start
    n = len(cases)
    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    # --- scorecard ---
    print(f"\n{'='*56}")
    print(f"RETRIEVAL EVAL  (k={args.k}, n={n})")
    print(f"  hit-rate@{args.k}: {hit_rate:.1%}  ({hits}/{n})")
    print(f"  MRR@{args.k}:      {mrr:.3f}")
    print(f"  elapsed:        {elapsed:.1f}s")
    print(f"{'='*56}")
    misses = [c for c in per_case if not c["hit"]]
    if misses:
        print(f"\nMISSES ({len(misses)}):")
        for m in misses:
            print(f"  {m['id']} ({m['ticker']}): labeled chunk not in top-{args.k}")

    # --- results file ---
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"retrieval_eval_{date.today().isoformat()}.md"
    with out.open("w") as f:
        f.write(f"# Retrieval Eval — {date.today().isoformat()}\n\n")
        f.write(f"- eval set: `{args.eval_set}` ({n} cases)\n")
        f.write(f"- k: {args.k}\n")
        f.write(f"- **hit-rate@{args.k}: {hit_rate:.1%}** ({hits}/{n})\n")
        f.write(f"- **MRR@{args.k}: {mrr:.3f}**\n")
        f.write(f"- elapsed: {elapsed:.1f}s\n\n")
        f.write("| id | ticker | rank | hit |\n|----|--------|------|-----|\n")
        for c in per_case:
            f.write(f"| {c['id']} | {c['ticker']} | "
                    f"{c['rank'] if c['rank'] else '—'} | "
                    f"{'✓' if c['hit'] else '✗'} |\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()