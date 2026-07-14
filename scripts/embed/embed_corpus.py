"""Delta-aware embedding: read chunks, embed with BGE-M3, load into the store.

Only filings not already in the store are embedded (delta by default), mirroring
the delta-chunking pattern. Embedding + loading happen per-filing, so a crashed
run resumes without redoing completed filings.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from scripts.embed.local_embedder import LocalEmbedder
from scripts.store.pgvector_store import PgVectorStore


def _load_chunks_by_accession(corpus_path) -> dict[str, list[dict]]:
    """Read corpus.jsonl and group chunks by accession_number."""
    by_accession: dict[str, list[dict]] = defaultdict(list)
    with Path(corpus_path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                by_accession[chunk["accession_number"]].append(chunk)
    return dict(by_accession)


def _empty_cache() -> None:
    """Free cached GPU memory (MPS or CUDA) if available; no-op on CPU."""
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_corpus(
    corpus_path="data/chunks/corpus.jsonl",
    force: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """Embed and load corpus chunks into the store, delta by default.

    force=True clears the store and re-embeds everything. Returns
    (filings_processed, chunks_added).
    """
    embedder = LocalEmbedder()
    store = PgVectorStore()

    # Dimension guardrail: embedder output must match the store's vector column.
    assert embedder.dimension == 1024, f"unexpected dim {embedder.dimension}"

    if force:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE chunks")
            conn.commit()
        if verbose:
            print("force=True: cleared the chunks table")

    by_accession = _load_chunks_by_accession(corpus_path)
    already = store.existing_accessions()
    todo = [acc for acc in by_accession if acc not in already]

    if verbose:
        print(f"{len(by_accession)} filings in corpus, {len(already)} already "
              f"embedded, {len(todo)} to process")

    filings_done = chunks_added = 0
    start = time.time()

    for acc in todo:
        chunks = by_accession[acc]
        ticker = chunks[0]["ticker"]
        year = chunks[0]["fiscal_year"]
        texts = [c["text"] for c in chunks]

        if verbose:
            elapsed = time.time() - start
            print(f"[{filings_done + 1}/{len(todo)}] embedding {ticker} {year} "
                  f"({len(texts)} chunks) | {chunks_added} done so far "
                  f"| {elapsed:.0f}s", flush=True)

        # show_progress draws a live per-batch bar inside this filing, so even
        # large filings show movement rather than appearing frozen.
        vectors = embedder.embed(texts, show_progress=verbose)
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        chunks_added += store.add(chunks)
        filings_done += 1

        # Release cached MPS/GPU memory between filings. PyTorch's MPS backend
        # does not free memory between encode() calls, so allocation creeps up
        # across filings and eventually OOMs; clearing the cache keeps it flat.
        _empty_cache()

    if verbose:
        print(f"\nDone: {filings_done} filings, {chunks_added} chunks embedded "
              f"in {time.time() - start:.0f}s")
    return filings_done, chunks_added
