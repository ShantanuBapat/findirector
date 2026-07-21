# FinDirector — Future Work

Deferred ideas and future-iteration enhancements — considered but parked for a
later version. Near-term active work and settled architectural decisions live
elsewhere; this is the "someday / vNext" roadmap. Newest entries on top.

Each entry: what it is, why it matters, the caveats, rough effort, and where the
idea came from.

---

## `research`: workflow vs agent (two query types, different handling)

**Date:** 2026-07-14 · **Origin:** Session 3.4 (conversation-flow design) · **Target:** v2 (workflow) / v3 (agent)

**Context.** The `research` action code covers all multi-document queries, but its
trigger conditions span two structurally different query types that should *not* be
handled the same way. The directive prompt's own triggers name both: "multiple
companies / multiple periods" (one type) and "multi-hop reasoning where one
retrieved fact informs the next query" (the other).

**The idea — split `research` by control-flow structure:**

*Type 1 — parallel/fixed multi-doc (deterministic workflow).* e.g. "Compare Apple
and Microsoft's 2023 operating margins." The retrieval steps are knowable up front:
retrieve for each named company/year, then synthesize. This is a **fan-out +
synthesize workflow** — a loop over a known set, then one generation call. No agent
needed.

*Type 2 — multi-hop / exploratory (genuine agent).* e.g. "Which of these companies
has the most supply-chain risk, and why?" The system must retrieve, read, and
**decide what to retrieve next based on what it found** — the path can't be
pre-planned because it depends on intermediate results. This is the real agent case:
a reasoning loop that plans its own retrieval sequence and stops when it has enough.

**The decision rule.** Does the sequence of retrievals depend on what earlier
retrievals return? No → workflow (Type 1). Yes → agent (Type 2). Forcing Type 1
through an agent adds nondeterminism, latency, and cost to solve what is essentially
a `for` loop.

**Why it matters.** This is the concrete boundary between "workflow" and "agent" in
FinDirector — and the justification for *why* the project uses both. Building all
`research` as an agent would be over-engineering; building none as an agent would
make the most impressive (multi-hop) queries impossible. Handling the two types
differently is the "judgment about when to use agents" the project is meant to
demonstrate.

**Roadmap fit.** v2 builds the Type 1 workflow (deterministic fan-out + synthesize —
covers the bulk of comparison queries). v3 introduces the Type 2 agent — this is
where LangGraph earns its place (multi-hop control flow, tool use, looping),
consistent with the "LangGraph for v3 agentic actions" decision. `research` thus
*evolves* workflow → agent as query difficulty grows.

**Rough effort.** Medium for Type 1 (fan-out retrieval + synthesis over the existing
RAG loop). Higher for Type 2 (agent framework, loop control, termination logic,
guardrails against runaway retrieval).

---

## Fiscal-year query mapping for non-December filers

**Date:** 2026-07-10 · **Origin:** Session 3.2 metadata extraction · **Target:** Week 3/4 (retrieval)

**Context.** `fiscal_year` is derived from CONFORMED PERIOD OF REPORT and follows
the filer's own convention. For Jan/Feb fiscal-year-ends this can surprise users:
WMT's filing with period 20260131 is tagged `fiscal_year=2026` (Walmart's "FY2026"),
and NVDA similarly reaches 2026. A user asking for "Walmart 2025" may mean the
period that ended Jan 2026, or the one that ended Jan 2025.

**The idea.** At retrieval time, when the directive model extracts `year: N`,
map it to the right filing(s) accounting for fiscal calendar — e.g. consider both
the `fiscal_year` label and the `period_of_report` date, or expand the year
filter by ±1 for known non-December filers. The full `period_of_report` (YYYYMMDD)
is already stored on every chunk as provenance to support this.

**Why it matters.** Prevents "correct-but-surprising" misses where a query for a
calendar year silently retrieves the wrong fiscal filing.

**Caveats.** Only affects the handful of non-December filers in the corpus. Low
urgency until retrieval is built; noting now so it is not discovered as a silent
miss later.

**Rough effort.** Low: a year-to-filing resolution step in the retrieval layer.

---

## Adaptive chunk size by section type

**Date:** 2026-07-10 · **Origin:** Session 3.1 · **Target:** post-v1 (tuning)

**Context.** v1 uses a uniform ~512-token target across all sections. But sections
differ in information density: financial statements (Item 8) are fact-dense and
favor tight, precise chunks; narrative sections (MD&A / Item 7, Risk Factors /
Item 1A) are discursive and favor larger chunks that preserve argument context.

**The idea.** Vary the target chunk size (and possibly overlap) by section type —
smaller for fact-dense/tabular sections, larger for narrative — via a per-Item
config mapping.

**Why it matters.** A uniform size compromises both ends: it's larger than ideal
for precise `lookup`/`compute` retrieval and smaller than ideal for
context-hungry `research`. Adaptive sizing could improve both at once.

**Caveats / why deferred.**
- Adds config complexity.
- The gain is an empirical claim — needs the Week 4 retrieval eval to confirm
  variable sizing actually beats uniform (measure, don't assume). Gains may be
  marginal.

**Rough effort.** Low-medium: parameterize the chunker by section, tune against
the retrieval eval.

---

## Expand corpus coverage (companies, years, filing types)

**Date:** 2026-07-10 · **Origin:** Session 3.1 planning · **Target:** post-v1

**Context.** v1 corpus is fixed: 20 tickers, 10-K only, 2022-2024. Expanding
coverage is a natural growth path, but "more companies" is really several
distinct axes with different payoffs.

**The idea — axes of expansion:**
- **Breadth (more tickers).** 20 → 100 → S&P 500. Widens the universe of
  answerable companies.
- **Depth (more years).** Extend back before 2022 to support longer-horizon
  trend and historical questions.
- **Filing types (beyond 10-K).** Add 10-Q (quarterly) and 8-K (material events).
  Notably, **adding 10-Q directly retires a real eval failure**: the Session 2.6
  failure analysis found the model tried to answer a Pfizer *10-Q* query that it
  had to decline only because the corpus is 10-K-only. With 10-Qs in scope, that
  becomes a legitimately answerable query.

**Demand-driven prioritization (the key refinement).** Don't expand arbitrarily —
expand toward the **most-requested-but-uncovered** companies. The mechanism is
already half-built: the Week 3 **corpus-boundary guard** fires a `decline`
whenever a user asks about something out of corpus (e.g. Netflix). If we **log
those out-of-corpus declines**, the aggregated log *becomes* a prioritized
expansion backlog automatically — the most-frequently-missed tickers are the ones
to add next. This is a self-improving loop:

    boundary guard fires -> log the miss -> aggregate -> top misses = next tickers

Implication for Week 3: build the boundary guard so it doesn't just refuse, but
**logs what it refused**, capturing demand signal from day one.

**Why it matters.** More coverage = more of the user's real questions are
answerable rather than declined. Demand-driven expansion ensures the coverage we
add is the coverage users actually want; the 10-Q axis converts a current
forced-decline into a real capability.

**Caveats / why deferred.**
- **Coupled to the corpus-boundary guard.** Every expansion moves the `decline`
  line. The boundary guard (Week 3) must be driven by the actual corpus manifest,
  not hardcoded to 20 tickers — so coverage changes don't require re-teaching the
  boundary each time. Build the guard manifest-driven (and instrumented to log
  misses) from the start.
- **Not free.** More filings → more chunks → more embedding cost, larger vector
  store, and possibly regenerating synthetic training data over the expanded
  ticker set so the directive model knows the wider universe.
- v1 should prove the loop on the focused 20-ticker corpus first, then scale.

**Rough effort.** Low-to-medium for breadth/depth (re-run the EDGAR downloader +
chunker over a bigger manifest). Higher for 10-Q/8-K (different document
structures need parser adjustments). Demand logging is a small add to the
boundary guard.

---

## Ingest high-value exhibits + XBRL to augment RAG

**Date:** 2026-07-10 · **Origin:** Session 3.1 corpus inspection · **Target:** post-v1 (v2/v3)

**Context.** Each SEC submission (`full-submission.txt`) contains the 10-K body
plus ~90 other documents (`PUBLIC DOCUMENT COUNT: 91` on the AAPL 2025 filing).
The v1 chunking pipeline selects only the `<TYPE>10-K` document and ignores the
rest. Some of that ignored content is genuinely valuable.

**The idea — two threads:**

*Thread A — selective high-value exhibit ingestion.* Not "read all 90 exhibits"
(most are legal boilerplate, material contracts, and individual XBRL fragments
that rarely answer a financial question). Instead, ingest the few high-signal
ones:
- Exhibit 21 — subsidiaries list (entity / ownership questions)
- Exhibit 99 — additional financial statements, press releases, guidance
- Exhibits 31/32 — SOX certifications (CEO/CFO sign-offs)

Add these as additional tagged chunks in the same vector store, with metadata
marking them as exhibit content (`item: exhibit_21`, etc.) so retrieval can
weight or filter them.

*Thread B — XBRL as a structured-fact source (higher leverage).* The 10-K body is
inline XBRL (iXBRL): every financial fact is machine-readable-tagged
(us-gaap concepts, values, periods). Rather than retrieving a chunk and hoping
the generator reads the table correctly, `lookup`/`compute` numeric queries could
resolve against the *structured* XBRL facts directly. This is potentially a whole
second retrieval **path**: structured-fact lookup for exact numbers, semantic RAG
for narrative. Likely more reliable for "what was revenue in 2023?" style queries
than HTML-table parsing.

**Why it matters.** Improves answer grounding and coverage. Thread B in
particular could sharply reduce numeric errors on `compute`/`lookup`, which is
the class of query where wrong numbers are most damaging for a financial tool.

**Caveats / why deferred.**
- Value is concentrated, not spread — "ingest everything" would add noise.
- Thread B (XBRL parsing) adds real ingestion complexity and a second retrieval
  path to design, route, and evaluate — scope creep for v1.
- v1 should prove the core directive-driven RAG loop first; this augments it.

**Candidate implementation.** `edgartools` — evaluated during Session 3.2 as the
parsing library. It offers native 10-K support and free inline-XBRL structured
facts (exactly Thread B), but was deferred to keep v1 lean (`sec-parser` chosen
instead). It becomes the natural tool if/when this XBRL path is built.

**Rough effort.** Medium. Thread A is a modest extension of the existing chunker.
Thread B is a larger design effort (structured-data store + a numeric-lookup
retrieval path + routing logic to choose it over semantic RAG).
