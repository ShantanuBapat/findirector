"""Embedding pipeline, split into two independently runnable stages:

  embed_to_file:  chunks (corpus.jsonl) -> embedded chunks (corpus_embedded.jsonl)
                  Compute-heavy; runs wherever the GPU is (Colab).
  load_from_file: embedded chunks -> pgvector store
                  I/O-light; runs wherever the database is (local).

The two are bridged by a file of embedded chunks, so embedding never needs
network access to the database (Colab GPU + local store).
"""

import json
import time
from pathlib import Path

from scripts.store.pgvector_store import PgVectorStore


def _empty_cache() -> None:
    """Free cached GPU memory (MPS or CUDA) if available; no-op on CPU."""
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_to_file(
    corpus_path="data/chunks/corpus.jsonl",
    out_path="data/chunks/corpus_embedded.jsonl",
    embedder=None,
    verbose: bool = True,
) -> int:
    """Embed every chunk in corpus_path, writing chunks+vectors to out_path.

    Writes one JSON object per line (the chunk dict plus an "embedding" key).
    Streams line-by-line so memory stays bounded. `embedder` defaults to
    LocalEmbedder but can be injected (e.g. a GPU embedder on Colab).
    Returns the number of chunks embedded.
    """
    if embedder is None:
        from scripts.embed.local_embedder import LocalEmbedder
        embedder = LocalEmbedder()
    assert embedder.dimension == 1024, f"unexpected dim {embedder.dimension}"

    corpus_path = Path(corpus_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(l) for l in corpus_path.open() if l.strip()]
    if verbose:
        print(f"embedding {len(chunks)} chunks -> {out_path}")

    written = 0
    start = time.time()
    with out_path.open("w") as out:
        # Embed in modest slices, clearing GPU cache between them, so memory
        # stays flat across the whole corpus.
        SLICE = 256
        for i in range(0, len(chunks), SLICE):
            batch = chunks[i:i + SLICE]
            texts = [c["text"] for c in batch]
            vectors = embedder.embed(texts, show_progress=False)
            for chunk, vector in zip(batch, vectors):
                chunk["embedding"] = vector
                out.write(json.dumps(chunk) + "\n")
            written += len(batch)
            _empty_cache()
            if verbose:
                elapsed = time.time() - start
                print(f"  {written}/{len(chunks)} chunks | {elapsed:.0f}s",
                      flush=True)

    if verbose:
        print(f"Done: {written} chunks embedded in {time.time() - start:.0f}s")
    return written


def load_from_file(
    embedded_path="data/chunks/corpus_embedded.jsonl",
    force: bool = False,
    batch_size: int = 500,
    verbose: bool = True,
) -> int:
    """Load embedded chunks from a file into the pgvector store.

    Delta by default: filings whose accession is already in the store are
    skipped. force=True truncates the store first (full rebuild). Returns the
    number of chunks inserted.
    """
    store = PgVectorStore()
    embedded_path = Path(embedded_path)

    if force:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE chunks")
            conn.commit()
        if verbose:
            print("force=True: cleared the chunks table")

    already = store.existing_accessions()
    if verbose and already:
        print(f"{len(already)} filings already loaded; loading delta only")

    inserted = 0
    buffer: list[dict] = []
    start = time.time()
    with embedded_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk["accession_number"] in already:
                continue
            buffer.append(chunk)
            if len(buffer) >= batch_size:
                inserted += store.add(buffer)
                buffer = []
                if verbose:
                    print(f"  inserted {inserted} | {time.time()-start:.0f}s",
                          flush=True)
    if buffer:
        inserted += store.add(buffer)

    if verbose:
        print(f"Done: {inserted} chunks loaded in {time.time() - start:.0f}s")
    return inserted
