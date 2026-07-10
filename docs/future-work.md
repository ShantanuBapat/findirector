# FinDirector — Future Work

Deferred ideas and future-iteration enhancements — considered but parked for a
later version. Near-term active work and settled architectural decisions live
elsewhere; this is the "someday / vNext" roadmap. Newest entries on top.

Each entry: what it is, why it matters, the caveats, rough effort, and where the
idea came from.

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
