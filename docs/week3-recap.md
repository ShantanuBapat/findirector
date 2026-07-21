# Week 3 Recap — The RAG Retrieval & Generation Pipeline
**Status:** complete (Sessions 3.1–3.4) · **Scope:** raw SEC filings → grounded, sourced answers

Week 3 turned the FinDirector corpus into a working retrieval-augmented question
answering system. A real query now flows end to end:

```
"What was Apple's R&D spending in fiscal 2023?"
  → directive params {AAPL, 2023, fact: "R&D spending"}
  → filtered vector retrieval (the R&D chunk ranks #1)
  → grounded generation
  → "$29,915 million ... (AAPL FY2023, mdna)"
```

This document recaps each step: **why we needed it**, **what breaks without it**,
and the **main code parts (inputs → outputs)**.

---

## 1. Chunking — filings into retrievable pieces

**Why we needed it.** A 10-K is a ~1–5 MB document. You cannot embed or retrieve a
whole filing as one unit — retrieval needs pieces small enough to be a precise
match for a question, but whole enough to carry meaning. Chunking is the transform
from "one giant document" to "many retrievable, tagged pieces."

**What breaks without it.** No chunking → nothing to embed at the right
granularity. Either you embed whole filings (retrieval returns a 200-page document,
useless) or you split blindly by size (severing sections and tables mid-figure,
destroying meaning). Retrieval quality collapses.

**Main code parts** (`scripts/chunk_filings.py`):
- `extract_primary_document(submission_text)` — **in:** raw SEC submission wrapper;
  **out:** the clean 10-K HTML body (unwrapped from the iXBRL container).
- `walk_sections(elements, metadata)` — **in:** parsed elements + filing metadata;
  **out:** section-tagged records (narrative accumulated per section; tables kept
  atomic as Markdown).
- `split_records(records)` — **in:** section records; **out:** final chunks, each
  ≤480 tokens of narrative (tables pass through atomic), with paragraph-level
  overlap.
- **Design:** section-aware + size-bounded + metadata-tagged. Tables stay whole so
  numbers never separate from their headers.

---

## 2. Delta-aware ingestion & persistence

**Why we needed it.** The corpus grows (new quarters, more tickers). Re-chunking all
60 filings every time you add one is fine at 60, absurd at 6,000. Delta processing
means only *new* filings are chunked; persistence means chunks survive to disk so
downstream steps don't re-chunk.

**What breaks without it.** No persistence → chunks vanish when the process exits;
every embedding run re-parses every filing. No delta → adding one filing reprocesses
the whole corpus. The pipeline doesn't scale past a demo.

**Main code parts** (`scripts/chunk_filings.py`):
- `accession_from_path(path)` — **in:** a filing path; **out:** its SEC accession
  number (the globally-unique, stable identity key). Costs no file I/O.
- `chunk_corpus(root, corpus_path, force)` — **in:** the filings root + output path;
  **out:** appends new filings' chunks to `corpus.jsonl`, skipping accessions
  already present. Per-filing append = crash-safe and resumable.
- **Result:** 18,742 chunks from 60 filings (20 tickers × 3 years), delta-aware,
  CLI-runnable (`python -m scripts.chunk_filings`).

---

## 3. Embedding — text into vectors

**Why we needed it.** Text can't be compared for *meaning* directly. Embedding maps
each chunk to a 1024-dim vector positioned so semantically similar text sits close
together — which is what makes "find the relevant chunk for this question" possible.

**What breaks without it.** No embeddings → no semantic search. You'd be left with
keyword matching, which misses paraphrases ("R&D spending" vs "research and
development expense") and can't rank by relevance.

**Main code parts** (`scripts/embed/`):
- `EmbeddingModel` (`base.py`) — the swappable interface: `dimension`,
  `embed(texts)`.
- `LocalEmbedder` (`local_embedder.py`) — **in:** list of texts; **out:** normalized
  1024-dim vectors, via BGE-M3. Cap at 1024 tokens; FP16 for speed.
- `embed_to_file` / `load_from_file` (`embed_corpus.py`) — split so embedding runs
  where the GPU is (Colab) and loading runs where the DB is (local), bridged by a
  file. **in:** `corpus.jsonl`; **out:** `corpus_embedded.jsonl` → pgvector rows.
- **The journey:** local MPS OOM'd repeatedly (7B-class attention memory on large
  tables); resolved by embedding on a Colab GPU in FP16 (a ~4× speedup that was the
  real fix), round-tripped through HuggingFace.

---

## 4. Vector store — holding & searching vectors

**Why we needed it.** The vectors need a home that supports two things at once: fast
nearest-neighbor search (find the closest vectors to a query) *and* metadata
filtering (restrict to a company/year). pgvector gives both inside Postgres.

**What breaks without it.** No store → nowhere to search. And without metadata
filtering, you can't constrain retrieval to the right company/year, and you lose the
corpus-boundary guard entirely.

**Main code parts** (`scripts/store/`):
- `VectorStore` (`base.py`) — swappable interface: `add`, `search`.
- `PgVectorStore` (`pgvector_store.py`) — **`add(chunks)`**: batch-insert embedded
  chunks. **`search(vec, k, filters)`**: cosine (`<=>`) ranking + SQL `WHERE`
  metadata filter, returns top-k with scores. **`existing_accessions()`**: the delta
  key for loading.
- **Infra:** pgvector in local Docker ($0), HNSW index built after load for fast
  search, secrets via `.env`. Migrates to RDS in Week 5.

---

## 5. Retrieval integration — directive to filtered search

**Why we needed it.** The directive model outputs `{company, year, fact_requested}`,
but nothing turned that into a filtered search + a decline decision. This is the
bridge that makes the directive *drive* retrieval.

**What breaks without it.** No bridge → the directive and store are islands. And with
no corpus-boundary guard, an out-of-corpus query (Netflix) returns empty context the
generator would hallucinate over.

**Main code parts** (`scripts/retrieval/retrieve.py`):
- `normalize_ticker(company)` — **in:** directive ticker (`BRK.B`); **out:** store
  convention (`BRK-B`). Reconciles the two data sources; without it, Berkshire
  queries wrongly decline.
- `_embedding_text(query, params)` — **in:** query + params; **out:** the distilled
  `fact_requested`/`concept` to embed (not the noisy full query). This moved the R&D
  chunk from rank 11 → rank 1.
- `retrieve(query, params, embedder, store, k)` — **in:** raw query + directive
  params; **out:** `{status: ok, chunks}` on a hit, or `{status: decline, ...}` on an
  out-of-corpus miss (which is also logged for demand-driven expansion).

---

## 6. Generation — context into grounded answers

**Why we needed it.** Retrieval finds the right passages; something has to *read* them
and write the answer. Generation turns retrieved chunks into a natural-language,
sourced response.

**What breaks without it.** No generation → the user gets raw chunks, not an answer.
And without the grounding rules, the model would answer from its own (stale,
un-auditable) training knowledge instead of the filings.

**Main code parts** (`scripts/generation/`):
- `Generator` (`base.py`) — swappable interface: `generate(query, chunks)`.
- `AnthropicGenerator` (`anthropic_generator.py`) — **in:** query + chunks; **out:** a
  grounded, cited answer. `_format_context` labels chunks by source; the system
  prompt enforces the rules: answer *only* from the excerpts (no hallucination), say
  so if the answer isn't present, cite inline as `(TICKER FY, section)`, refuse
  investment advice.
- **Backend:** Anthropic API for v1 (proven, fast); self-hosted vLLM planned for
  Week 5 — a drop-in swap thanks to the interface.

---

## The big picture — how it connects

```
user query
   │
   ▼
[directive model]  → {action_code, params}      (Week 2; API-routed in v1)
   │  (lookup)
   ▼
[retrieve()]       → normalize + filter + embed fact → vector search
   │                  ├─ hit    → chunks
   │                  └─ miss   → decline (+ log)
   ▼
[generate()]       → grounded, sourced answer
   │
   ▼
answer
```

Each stage is behind a **swappable interface** (`EmbeddingModel`, `VectorStore`,
`Generator`) — so embedding backend, store, and generation model can each change
without touching callers. This is what makes the Week 5 migrations (RDS, vLLM) clean.

## Key decisions & lessons

- **The directive-driven architecture pays off twice.** The directive's structured
  output drives *both* routing *and* retrieval precision (`fact_requested` as a
  ranking signal, rank 11 → 1).
- **Relative ranking beats absolute score.** Embedding the distilled fact *lowered*
  the R&D chunk's raw score but raised it to rank 1, because it lowered every other
  chunk's score more.
- **Attention memory scales with sequence length squared.** Big atomic tables +
  batching = OOM. Resolved with a length cap, FP16, and moving to a real GPU.
- **Separate compute from storage.** Embedding (GPU) and loading (DB) as independent
  stages, bridged by a file, is both the fix for the Colab/local split and better
  architecture.
- **Normalize at integration boundaries.** Same ticker, two source conventions
  (`BRK.B` vs `BRK-B`) — a classic data-integration seam, fixed where the systems
  meet.

## What's deferred (and why)

- **3.5 full router integration** — waits until `compute` (calculator) and `research`
  (multi-doc synthesis) exist, so the end-to-end test exercises real branches, not
  stubs. The `lookup` path is already proven.
- **Self-hosted directive & generation models (vLLM)** — v1 uses the Anthropic API;
  self-hosting is a Week 5 production task (the interfaces make it a drop-in swap).
- **Hybrid/sparse retrieval, giant-table embedding, corpus expansion** — noted in
  `docs/future-work.md`, demand-driven.
