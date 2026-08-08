# FinDirector

> A directive-driven RAG system for SEC filings analysis — demonstrating the multi-LLM orchestration pattern applied to a regulated industry, with end-to-end MLOps.

**Status:** v1 complete (directive model + full RAG lookup loop, working end-to-end) · v2 in progress (serving API, evaluation harness, then AWS deployment).

---

## What it does

FinDirector answers natural-language questions about public companies from their SEC 10-K filings — with **sourced, grounded answers**, or an explicit **decline** when a question crosses into investment advice, prediction, or out-of-scope territory.

A real query, running end to end today:

```
Q: "What was Apple's research and development spending in fiscal 2023?"

→ directive model classifies: lookup {company: AAPL, year: 2023, fact_requested: "R&D spending"}
→ retrieval filters to AAPL FY2023 chunks, ranks by the distilled fact
→ generation answers only from the retrieved excerpts:

A: "Apple's R&D spending in fiscal 2023 was $29,915 million (~$29.9 billion),
    an increase of 14% over fiscal 2022, and about 8% of net sales.
    (AAPL FY2023, mdna)"
```

Every fact is cited to its filing; nothing is answered from the model's own memory.

---

## Current results (v1)

| What | Result |
|---|---|
| **Directive model accuracy** | **94.87%** (148/156 held-out test set) — Qwen 2.5 7B + LoRA |
| **RAG corpus** | 18,742 chunks from 60 filings (20 tickers × 3 most-recent 10-Ks) |
| **End-to-end `lookup`** | Working: query → classify → filtered retrieval → grounded, cited answer |
| **Retrieval precision win** | Embedding the directive's distilled `fact_requested` moved the target chunk from rank 11 → rank 1 |

The directive model's per-code F1 ranges from 0.91 (`clarify`) to 1.00 (`compute`, `smalltalk`). A per-example failure audit found that **only 2 of 8 errors are genuine model errors** — the rest are a token-budget truncation artifact (2) and arguable label noise where the model's choice was defensible (4). Full analysis in [`results/eval_2026-07-09.md`](results/eval_2026-07-09.md).

---

## What makes it different

### 1. Directive-driven architecture

Rather than one model doing everything, responsibility is split:

- A **fine-tuned directive model** (Qwen 2.5 7B + LoRA) decides *how* to handle each query. It emits a structured object — `{action_code, params, reasoning}` — not user-facing text. The `action_code` is one of seven routing decisions; `params` carries the extracted specifics (company, year, the fact being asked).
- A **separate generation model** consumes the retrieved context and writes the grounded answer.

This separation buys **auditability** (the routing decision is structured and inspectable, separate from the wording), **safety** (policy can be examined independently of presentation), and **iteration speed** (improving routing doesn't require retraining the generator). In a regulated domain, you can answer a regulator's *"how did the system decide not to give investment advice?"* with a logged directive, not an opaque text generation.

**A second, less obvious payoff:** the directive's structured output improves *retrieval*, not just routing. Because it already extracted the semantic target (`fact_requested`), retrieval embeds *that* — not the noisy full query — which sharply improves ranking (the rank 11 → 1 result above). The directive-driven design pays off twice.

### 2. Safety and corpus boundaries as first-class concerns

- The taxonomy includes explicit `decline` codes for investment advice, predictions, and out-of-scope queries. In v1, these grounding rules (answer only from excerpts, cite sources, refuse investment advice) are **enforced in the generation model's system prompt**; a *dedicated* safety-classifier layer is planned for v4.
- A **corpus-boundary guard** in retrieval returns an explicit decline — rather than empty context the generator might hallucinate over — when a query falls outside the 20-ticker / 10-K corpus. Every such miss is **logged**, turning real out-of-corpus demand into a prioritized corpus-expansion backlog.

### 3. Evolves from orchestration to agents

- **v1–v2:** deterministic orchestrated workflow — predictable, auditable.
- **v3:** the `compute` and `research` actions become agentic — tools, reasoning loops, multi-hop retrieval — where the problem genuinely needs it.
- **v4:** production hardening — observability, traces, a dedicated safety layer, cost optimization.

This demonstrates judgment about *when* to use a workflow versus an agent — the `research` code, for instance, splits into a deterministic fan-out-and-synthesize workflow (multiple named companies) versus a genuine agent (multi-hop, where each retrieval depends on the last). See [`docs/future-work.md`](docs/future-work.md).

---

## Architecture

```
User query
   │
   ▼
Directive model  ──►  { action_code, params, reasoning }     Qwen 2.5 7B + LoRA
   │                                                          (94.87% on held-out test)
   ├─ smalltalk ─► canned social reply                       ┐
   ├─ meta      ─► static help content                       │ no retrieval
   ├─ clarify   ─► return the clarifying question             │
   ├─ decline   ─► canned safe refusal                        ┘
   │
   ├─ lookup    ─► retrieve (single-doc, filtered) ─► generate     ◄── BUILT (v1)
   ├─ compute   ─► retrieve ─► calculator ─► generate              ◄── planned (v3)
   └─ research  ─► multi-doc retrieve ─► synthesis ─► generate     ◄── planned (v3)
                              │
                              ▼
                   Generation model  ──►  grounded, cited answer
                   (Anthropic API in v1; self-hosted Qwen via vLLM in v2/Week 5)
```

Every heavy component sits behind a swappable interface — `EmbeddingModel`, `VectorStore`, `Generator` — so the embedding backend, the store, and the generation model can each change without touching calling code. This is what makes the planned migrations (local pgvector → RDS; Anthropic API → self-hosted vLLM) drop-in swaps rather than rewrites.

---

## How the RAG pipeline works

The `lookup` path — the fully built v1 loop — in six stages, each behind a clean input → output boundary:

1. **Chunking** (`scripts/chunk_filings.py`) — raw SEC submission → section-aware, size-bounded, metadata-tagged chunks (tables kept atomic so figures never separate from headers). *60 filings → 18,742 chunks.*
2. **Embedding** (`scripts/embed/`) — chunks → normalized 1024-dim vectors via **BGE-M3**. Split into a GPU-side `embed_to_file` and a DB-side `load_from_file`, bridged by a file (so embedding runs on a Colab GPU, loading runs locally).
3. **Vector store** (`scripts/store/`) — **pgvector** (Postgres) with an HNSW index; supports cosine similarity *and* metadata filtering in one query.
4. **Retrieval** (`scripts/retrieval/retrieve.py`) — directive `params` → normalized ticker + year filters → filtered vector search on the distilled `fact_requested`. Returns retrieved chunks, or a corpus-boundary decline (logged).
5. **Generation** (`scripts/generation/`) — retrieved chunks → grounded, source-cited answer, constrained by a system prompt (answer only from excerpts; say so if absent; cite as `(TICKER FY, section)`; refuse investment advice).

The directive model itself (Qwen 2.5 7B + LoRA) is trained and evaluated by `scripts/train_directive_model.py` and `scripts/evaluate_directive_model.py`, on synthetic data generated and labeled by the pipeline in `scripts/` + `prompts/`.

---

## Action code taxonomy

The directive model classifies each query into exactly one of seven codes. **The decision boundary is the retrieval pattern, not topic complexity** — a conceptually deep single-company question is still `lookup`.

| Code | When | Routes to | Built? |
|------|------|-----------|--------|
| `smalltalk` | Greetings, thanks, low-content | Canned reply, no retrieval | ✅ v1 |
| `meta` | Questions *about* FinDirector | Static help, no retrieval | ✅ v1 |
| `lookup` | One fact/concept from ONE company's filing | RAG → generation | ✅ v1 |
| `compute` | Arithmetic on retrieved values (one company) | RAG → calculator → generation | 🔜 v3 |
| `research` | Multi-company, multi-period, or multi-hop | Multi-doc RAG → synthesis | 🔜 v3 |
| `clarify` | Ambiguous — missing company/year/reference | Return clarifying question | ✅ v1 |
| `decline` | Investment advice, prediction, out-of-scope | Canned safe refusal | ✅ v1 |

```
single doc + single fact         → lookup
single doc + arithmetic          → compute
multi doc OR multi time period   → research
```

### A few examples

- **`lookup`** — "What was Apple's R&D spending in fiscal 2023?" · "What does Microsoft mean by 'commercial cloud'?"
- **`compute`** — "What's Apple's R&D as a percentage of revenue for 2023?"
- **`research`** — "Compare Apple and Microsoft's 2023 operating margins." · "How has Tesla's gross margin changed from 2020 to 2024?"
- **`clarify`** — "What were earnings last quarter?" *(which company?)*
- **`decline`** — "Should I buy AAPL?" *(investment advice)* · "What's the weather?" *(out of scope)*

---

## Tech stack

| Layer | v1 (built) | v2 / production (planned) |
|-------|-----------|---------------------------|
| **Directive model** | Qwen 2.5 7B + LoRA (fine-tuned, 94.87%) | AWQ INT4 quantization for serving |
| **Embedding** | BGE-M3 (1024-dim), `sentence-transformers` | scaled GPU / managed embedding |
| **Vector store** | pgvector (Postgres 17) in Docker, HNSW index | pgvector on RDS |
| **Generation** | Anthropic API (`claude-sonnet-4-5`), grounded via system prompt | self-hosted Qwen via **vLLM** on EKS |
| **Chunking / parsing** | `sec-parser`, `tiktoken` | — |
| **Serving** | *(v2)* | FastAPI + SSE streaming |
| **Orchestration** | Docker Compose (local) | EKS + Karpenter (GPU autoscaling) |
| **IaC / CI-CD** | *(Week 5)* | Terraform; GitHub Actions with OIDC |
| **Observability** | *(Week 5)* | Prometheus + Grafana; Langfuse traces (v3+) |
| **Eval** | *(v2)* | ConvFinQA + custom LLM-as-judge |

Model and dataset artifacts are published on Hugging Face under `AlHindi/` (`findirector-directive-lora`, `findirector-splits`, `findirector-corpus`).

---

## Roadmap

- **v1 — Orchestrated workflow (Weeks 1–3) ✅ complete.** Directive model fine-tuned (94.87%); full RAG `lookup` loop working end-to-end.
- **v2 — Serving & evaluation (Weeks 4–5).** FastAPI + SSE; orchestrator routing all seven codes; eval harness (ConvFinQA + LLM-as-judge); observability; cost analysis; then AWS deployment (RDS, EKS, vLLM, Terraform, OIDC, Prometheus/Grafana).
- **v3 — Agentic extension (Weeks 6–7).** `compute` and `research` become agentic components with tools (SEC search, calculator, macro data) and multi-hop reasoning — a hybrid workflow-plus-agent architecture.
- **v4 — Production hardening (Week 8).** Dedicated safety classifier, Langfuse tracing, regulatory risk assessment, red-team suite, demo recording.

---

## Repository structure

```
findirector/
├── README.md
├── docker-compose.yml            # local pgvector database
├── requirements.txt
├── .env.example                  # config template (.env is gitignored)
├── docs/
│   ├── design/chunking.md        # chunking strategy + rationale
│   ├── week3-recap.md            # the RAG pipeline, step by step
│   └── future-work.md            # deferred ideas (incl. research workflow-vs-agent)
├── prompts/
│   ├── directive_labeler.py      # the 7-code classification prompt
│   └── synthetic_query_generator.py
├── scripts/
│   ├── download_filings.py       # SEC EDGAR ingest
│   ├── chunk_filings.py          # section-aware chunking
│   ├── embed/                    # EmbeddingModel interface + BGE-M3 + file-based stages
│   ├── store/                    # VectorStore interface + pgvector
│   ├── retrieval/                # directive → filtered search + corpus-boundary guard
│   ├── generation/               # Generator interface + Anthropic backend
│   ├── generate_synthetic_queries.py / label_synthetic_queries.py / build_splits.py
│   ├── train_directive_model.py / evaluate_directive_model.py / training_config.py
│   └── push_*_to_hf.py           # publish artifacts to Hugging Face
├── notebooks/                    # Colab GPU notebooks (train / embed / eval)
├── results/                      # evaluation reports
└── exploration/                  # one-off diagnostic + verification probes
```

---

## Evaluation, safety, and cost

*Full harnesses land in v2/v4; current state:*

- **Directive classification** — measured: **94.87%** on a 156-example held-out set, with per-code precision/recall/F1 and a per-example failure audit ([`results/`](results/)).
- **End-to-end answer quality** — planned for v2: ConvFinQA reference set, exact-match on numeric answers + LLM-as-judge on free text.
- **Safety / red teaming** — planned for v4: ~30 adversarial cases (prompt injection incl. via RAG documents, jailbreaks toward investment advice, out-of-scope baiting, citation-hallucination probes), with pass/fail rates documented.
- **Cost analysis** — planned for v2: per-query cost self-hosted on EKS vs. Bedrock vs. OpenAI, with break-even by daily volume.

---

## Running it locally

```bash
# 1. Configure secrets/config
cp .env.example .env        # then fill in POSTGRES_* and your ANTHROPIC_API_KEY

# 2. Start the local pgvector database
docker compose up -d

# 3. (Python env)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The corpus (`corpus.jsonl`) and embedded vectors (`corpus_embedded.jsonl`) are published on Hugging Face (`AlHindi/findirector-corpus`); the embedded file loads into pgvector via `scripts/embed/embed_corpus.py::load_from_file`.

---

## Why "directive-driven"?

Using one model to *decide* and another to *generate* gives, for regulated financial services specifically:

- **Structured, verifiable decisions** — the directive is a constrained object, not free text.
- **Separation of policy and presentation** — the same `decline` decision can be presented differently without changing the policy logic.
- **Auditability** — every routing decision is logged with its reasoning, separately from the generation. This is what lets you answer *"how did the system decide not to give investment advice?"* with evidence.

---

## License

MIT — see [LICENSE](LICENSE).
