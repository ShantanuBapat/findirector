"""Retrieval integration: map a directive to filtered vector search.

Bridges the directive model's output ({company, year, ...}) to the vector
store, applying the corpus-boundary guard: a query for a company/year not in
the corpus returns a decline signal (and is logged) rather than empty context
the generator might hallucinate over.
"""

import json
import time
from pathlib import Path

# Where corpus-boundary misses are logged (for demand-driven expansion).
MISS_LOG = Path("data/retrieval_misses.jsonl")


def normalize_ticker(company: str) -> str:
    """Reconcile the directive's ticker convention with the store's.

    The directive (trained on LLM-generated data) emits e.g. 'BRK.B'; the store
    (sourced from SEC EDGAR) uses 'BRK-B'. Normalize separators to the store's
    hyphen convention. Same-entity/different-format is a classic integration
    seam; we normalize at the boundary.
    """
    return company.strip().upper().replace(".", "-").replace("/", "-")


def _log_miss(query: str, filters: dict) -> None:
    """Record a corpus-boundary miss: what was asked but isn't in the corpus."""
    MISS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MISS_LOG.open("a") as fh:
        fh.write(json.dumps({
            "ts": time.time(), "query": query, "filters": filters,
        }) + "\n")


def retrieve(query: str, params: dict, embedder, store, k: int = 5) -> dict:
    """Retrieve filing chunks for a directive-classified query.

    params: the directive model's params, e.g. {"company": "AAPL", "year": 2023}.
      - company -> ticker filter (normalized).
      - year -> fiscal_year filter, applied only when present (null => all years
        for that company; ranking sorts relevance).
    Returns a dict:
      {"status": "ok", "chunks": [...]}                on a hit, or
      {"status": "decline", "reason": ..., "filters": ...}  on a corpus-boundary
      miss (empty result). The decline signal is what the generation layer uses
      to refuse rather than fabricate.
    """
    filters: dict = {}
    company = params.get("company")
    if company:
        filters["ticker"] = normalize_ticker(company)
    year = params.get("year")
    if year is not None:
        filters["fiscal_year"] = int(year)

    query_vec = embedder.embed([query])[0]
    chunks = store.search(query_vec, k=k, filters=filters or None)

    if not chunks:
        _log_miss(query, filters)
        return {
            "status": "decline",
            "reason": "no matching filings in corpus",
            "filters": filters,
        }

    return {"status": "ok", "chunks": chunks}
