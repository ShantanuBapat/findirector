# FinDirector

> A directive-driven RAG system for SEC filings analysis, demonstrating the multi-LLM orchestration pattern applied to a regulated industry.

**Status:** v0 (design phase). Architecture and taxonomy finalized; implementation begins Week 2.

---

## What This Project Does

FinDirector answers natural-language questions about public companies using their SEC filings (10-K annual reports). You can ask things like:

- *"What were Apple's main business risks in their 2023 10-K?"*
- *"How has Microsoft's gross margin trended over the last 5 years compared to Apple?"*
- *"What does Tesla mean by 'gigafactory' in their filings?"*

And get sourced, accurate answers — or an explicit decline if the question crosses into investment advice, predictions, or out-of-scope territory.

## What Makes It Different

Three architectural choices distinguish FinDirector from typical RAG chatbots:

### 1. Directive-driven architecture

Instead of one large model doing everything, two models split responsibility:

- **A fine-tuned directive model (Qwen 2.5 7B)** decides *how* to handle each query. It outputs a structured action code — like `lookup`, `compute`, `research`, or `decline` — rather than user-facing text.
- **A separate generation model** consumes the directive plus retrieved context and writes the actual response.

This separation gives:
- **Reliability** — the directive's output is structured and auditable
- **Safety** — policy decisions can be inspected independently from response wording
- **Faster iteration** — improving the policy doesn't require retraining the generator

This pattern is adapted from production multi-LLM systems used in regulated domains, where separating *decision-making* from *content generation* is essential for compliance and verifiability.

### 2. Regulatory safety as first-class architecture

Financial advice is regulated. FinDirector treats this as a design constraint rather than a guardrail:

- The directive taxonomy includes explicit `decline` codes for investment advice, predictions, and out-of-scope queries
- A dedicated safety classifier sits between generation and the user, blocking outputs that could be construed as financial advice
- All responses cite their SEC filing sources

### 3. Evolves from orchestration to agents

The project ships in four versions, demonstrating different production patterns:

- **v1-v2:** Pure orchestrated workflow — predictable, auditable
- **v3:** Selected actions (`research`, `compute`) become agentic — tools, reasoning loops, multi-hop retrieval
- **v4:** Production hardening — full observability, traces, safety, cost optimization

This evolution shows judgment about *when* to use workflows versus agents, and *how* to combine them — a question every modern LLM system architect has to answer.

---

## Architecture

```
User query
   ↓
[Intent classifier — small, fast]
   ↓
[Directive model — Qwen 2.5 7B fine-tuned with LoRA]
   ├─→ outputs: { action_code, params, recommendation }
   ↓
[Action router]
   ├─→ smalltalk → canned response, skip pipeline
   ├─→ meta      → static help content
   ├─→ lookup    → RAG (single doc) → generation
   ├─→ compute   → RAG + calculator → generation         [agent in v3]
   ├─→ research  → RAG (multi-doc) → synthesis           [agent in v3]
   ├─→ clarify   → return clarifying question, no generation
   └─→ decline   → safety classifier → canned decline
   ↓
[Generation model — Qwen 7B base or Claude via Bedrock]
   ↓
[Safety classifier — blocks investment-advice patterns]
   ↓
Final response (streamed via SSE)
```

---

## Action Code Taxonomy

The directive model classifies each user turn into exactly one of seven codes. The choice of code determines routing through the system.

| Code | When | Routes To | Agentic in v3? |
|------|------|-----------|----------------|
| `smalltalk` | Greetings, thanks, low-content messages | Canned response, skip pipeline | No |
| `meta` | Questions about FinDirector itself | Static help content | No |
| `lookup` | Single-document, single-fact retrieval | RAG → generation | No |
| `compute` | Retrieval + arithmetic on numerical data | RAG → calculator → generation | **Yes** |
| `research` | Multi-document, multi-hop, or multi-time-period reasoning | RAG (multi-doc) → synthesis | **Yes** |
| `clarify` | Query is ambiguous (which company? which year?) | Return clarifying question, no generation | No |
| `decline` | Investment advice, prediction, or out-of-scope | Safety classifier → canned decline | No |

### Classification Rule of Thumb

```
single doc + single fact         → lookup
single doc + arithmetic          → compute
multi doc OR multi time period   → research
```

Topic complexity does not determine the code — *retrieval pattern* does.

### Examples Per Code

**`smalltalk`**
- "Hi, are you online?"
- "Thanks!"
- "Got it, that's helpful."

**`meta`**
- "What can you do?"
- "What's the difference between a 10-K and a 10-Q?"
- "What's your knowledge cutoff?"

**`lookup`**
- "What was Apple's R&D spending in fiscal 2023?"
- "What does Microsoft mean by 'commercial cloud' in their filings?"
- "What's Tesla's stated mission?"

**`compute`**
- "What's Apple's 3-year average free cash flow?"
- "Calculate Microsoft's R&D as a percentage of revenue for 2023."
- "Average those two numbers you just gave me." *(stateful — uses prior turn context)*

**`research`**
- "How has Microsoft's gross margin trended over the last 5 years compared to Apple?"
- "Compare Tesla's debt levels from 2021 to 2024."
- "How has Apple's risk disclosure changed across the past 3 annual reports?"

**`clarify`**
- "Tell me about earnings." *(whose? which year?)*
- "How did the company do last quarter?" *(which company?)*
- "What are the risks?" *(of what?)*

**`decline`** (with sub-reasons in `params`)
- *Investment advice:* "Should I buy AAPL?", "Is Tesla overvalued?"
- *Prediction:* "Will the market crash this year?", "What will Apple's stock be in 2027?"
- *Out of scope:* "What's the weather?", "Help me write a Python script."

### Edge Cases Documented

A few cases that revealed important architectural decisions:

- **Skeptical user** ("I don't think that number is right") — code is still `lookup`, but the directive's `params` flag re-verification mode. The generation model cites sources explicitly when this flag is set.
- **Conversational state** ("average those two numbers you just gave me") — code is `compute`. The directive model classifies; the generation model resolves "those two numbers" from conversation history. Stateless directive, stateful generation.
- **Definition questions about a single company** ("what does MSFT mean by X") — `lookup`, not `research`. Single company + single concept = retrieval, regardless of conceptual depth.

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Model** | Qwen 2.5 7B + LoRA adapter | Open-weights, runs on a single L4 GPU when quantized; prior fine-tuning experience |
| **Quantization** | AWQ INT4 | ~4× memory reduction, modest quality loss |
| **Serving** | vLLM in Docker | Industry standard for high-throughput LLM inference (PagedAttention, continuous batching) |
| **Orchestration** | EKS with Karpenter | GPU-aware autoscaling, real production pattern |
| **RAG corpus** | SEC EDGAR 10-K filings (20 S&P 500 companies, multiple years) | Authoritative source, public domain |
| **Vector store** | OpenSearch Serverless or pgvector on RDS | TBD by cost; pgvector likely for personal-scale |
| **API** | FastAPI with SSE streaming | Standard, supports token-streaming |
| **IaC** | Terraform end-to-end | Industry standard, easier than CDK for portable patterns |
| **CI/CD** | GitHub Actions with OIDC federation | No long-lived AWS keys; modern best practice |
| **Deployment** | Helm + ArgoCD | GitOps pattern; reflects how production teams work |
| **Observability** | Prometheus + Grafana; Langfuse for LLM traces in v3+ | Industry-standard for both infra and LLM observability |
| **Eval** | ConvFinQA + custom LLM-as-judge harness | Established financial QA benchmark + project-specific evals |

---

## Evolution Roadmap

### v1 — Orchestrated workflow (Weeks 2-3)
Single-shot Q&A. Pure deterministic orchestration: intent classifier → directive model → action router → RAG → generation → safety check.

**Demonstrates:** the Lorebot pattern transferred to finance; multi-LLM coordination; fine-tuned directive model; baseline RAG.

### v2 — Evaluation and polish (Weeks 4-5)
Comprehensive eval harness, observability dashboards, cost analysis, README polish, demo recording.

**Demonstrates:** rigor — what separates AI engineering from prompt-hacking.

### v3 — Agentic extension (Weeks 6-7)
`compute` and `research` actions become agentic components. The agents use tools (SEC search, calculator, FRED API for macro data) and can take multiple internal steps before returning to the orchestrator.

**Demonstrates:** judgment about when to add agentic behavior; hybrid workflow + agent architecture; trajectory eval.

### v4 — Production hardening (Week 8)
Full Langfuse tracing, safety hardening, regulatory risk assessment doc, Loom recording, public release.

**Demonstrates:** production thinking; compliance-aware engineering.

---

## Repository Structure (planned)

```
findirector/
├── README.md                           # this file
├── docs/
│   ├── architecture.md                 # detailed architecture
│   ├── action-codes.md                 # full taxonomy with examples
│   ├── eval-methodology.md             # how we measure quality
│   ├── safety-and-red-teaming.md       # adversarial testing
│   └── cost-analysis.md                # $/1k tokens, comparisons
├── infra/                              # Terraform
│   ├── vpc/
│   ├── eks/
│   ├── ecr/
│   └── opensearch/
├── src/
│   ├── directive_model/                # fine-tuning + inference
│   ├── retrieval/                      # RAG pipeline
│   ├── generation/                     # response generation
│   ├── safety/                         # safety classifier
│   ├── api/                            # FastAPI + SSE
│   └── agents/                         # v3 agentic components
├── data/                               # gitignored
│   ├── filings/                        # downloaded 10-Ks
│   └── eval/                           # eval sets
├── notebooks/                          # exploration
├── tests/
├── .github/
│   └── workflows/                      # CI/CD
├── docker/
└── helm/                               # Kubernetes manifests
```

---

## Safety & Red Teaming

*[Placeholder — populated in v2]*

The system will be evaluated against ~30 adversarial test cases covering:

- Prompt injection (direct and via RAG documents)
- Jailbreak attempts targeting investment-advice generation
- Out-of-scope baiting
- Citation hallucination probes
- Bias probes (identical queries across different companies/sectors)
- Conversational state attacks (gradual scope drift across turns)

Pass/fail rates and failure-mode analysis will be documented here.

---

## Eval Methodology

*[Placeholder — populated in v2]*

Three layers of evaluation:

1. **Directive classification accuracy** — does the directive model assign the correct code? (Custom labeled set of 200+ queries)
2. **End-to-end answer quality** — measured against ConvFinQA reference answers using exact-match for numerical answers and LLM-as-judge for free-text
3. **Safety regression** — adversarial test suite, must pass 100% on critical categories (investment advice, predictions)

---

## Cost Analysis

*[Placeholder — populated in v2]*

Will compare per-query cost of:
- FinDirector self-hosted on EKS
- Equivalent capability via Bedrock Claude
- Equivalent capability via OpenAI API

With break-even analysis at different daily query volumes.

---

## Why "Directive-Driven"?

The architecture pattern of using one model to *decide* and another model to *generate* shows up in several recent production systems and research directions:

- **Structured decision-making** — the directive's output is a constrained JSON object, not free text. This is verifiable.
- **Separation of policy and presentation** — the same `decline` decision can be presented differently for different audiences without changing the policy logic.
- **Auditability** — every decision is logged with the directive's reasoning, separately from the generation. Critical for regulated industries.

For financial services specifically, this pattern lets you answer the regulator's question: *"How did the system decide not to give investment advice?"* with a logged directive output and the model's reasoning, rather than a single opaque text generation.

---

## Status

- [x] Architecture and action code taxonomy finalized (Week 1)
- [x] AWS infrastructure baseline (IAM Identity Center, billing alarms, GPU quota)
- [x] Repository initialized with `.gitignore` and design documentation
- [ ] SEC 10-K corpus downloaded (Week 2)
- [ ] Synthetic training data for directive model generated (Week 2)
- [ ] Directive model fine-tuned (Week 2-3)
- [ ] RAG pipeline implemented (Week 3)
- [ ] End-to-end v1 functional (end of Week 3)

---

## License

Private project; license TBD when made public.

## Acknowledgments

Architecture pattern inspired by multi-LLM orchestration systems used in conversational AI for regulated domains, including mental health support systems where separating directive policy from response generation is essential for safety and auditability.
