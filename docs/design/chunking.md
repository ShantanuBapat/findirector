# Chunking Design — FinDirector RAG Corpus

**Status:** decided (Session 3.1) · **Implementation:** Session 3.2

How the 20-ticker / 10-K corpus is turned into retrievable, tagged chunks for the
RAG pipeline. This is the spec the ingestion pipeline (3.2) implements.

## Corpus

- 20 tickers, 10-K only, fiscal years 2022-2024 (3 filings each, 60 total).
- Stored at `data/raw/sec_edgar/sec-edgar-filings/<TICKER>/10-K/<accession>/full-submission.txt`.
- Format: full SEC submission wrapper (`<SEC-HEADER>` + multiple `<DOCUMENT>`
  blocks). The 10-K body is inline XBRL (iXBRL) HTML — structure and tables
  recoverable (the good case).

## Strategy

Section-aware + size-bounded + metadata.

- **Section-aware:** split on 10-K Item boundaries first (Item 1 Business, 1A Risk
  Factors, 7 MD&A, 8 Financial Statements, ...), so a chunk never straddles two
  sections.
- **Size-bounded:** within a section, sub-split to a token target.
- **Metadata:** every chunk is tagged so retrieval can filter before ranking.

Rejected alternatives: fixed-size (structure-blind — severs sections and tables);
recursive-only (respects paragraphs but not document structure). Section-aware
wins because 10-Ks have a rigid, regular Item structure.

## Parse pipeline

1. Read `full-submission.txt`.
2. Unwrap the submission: select the `<DOCUMENT>` with `<TYPE>10-K` (SEQUENCE 1);
   ignore the ~90 other documents (exhibits, XBRL fragments).
3. Extract metadata from `<SEC-HEADER>` (authoritative): submission type
   (`filing_type`), CONFORMED PERIOD OF REPORT (`fiscal_year` — use the
   period-of-report date, not the filing date), company name / CIK; ticker comes
   from the directory path.
4. Parse the iXBRL HTML body.
5. Detect Item / section boundaries.
6. Chunk each section (parameters below), tagging every chunk with metadata.

**Tooling (chosen in 3.2): `sec-parser` 0.58.1.** Verified empirically against
real filings. Two findings shaped the approach:

- The released 0.58.1 has **no `Edgar10KParser`** (it exists only in the dev
  docs); only `Edgar10QParser` ships. `Edgar10QParser` extracts elements
  (text, tables, titles) from 10-Ks fine, but its `section_type` classifier
  applies the **10-Q** taxonomy and mislabels 10-K items, and it types item
  headings inconsistently (some `TopSectionTitle`, some `TitleElement`).
- Therefore we **detect sections ourselves**: scan the flat element list and
  treat any element whose text matches an item heading (via `identify_section`,
  an item-number → section-name map we own) as a section boundary. We do **not**
  use `sec-parser`'s `TopSectionTitle`/`section_type`. The resulting
  `InvalidTopSectionIn10Q` warnings are suppressed deliberately.

See `exploration/` for the probe scripts that established this.

The submission-unwrap step (extracting the `<TYPE>10-K` HTML from
`full-submission.txt`) is ours — `sec-parser` expects the HTML body. It also
strips the iXBRL `<XBRL>`/`<?xml?>` prefix so the parser receives clean HTML.

## Chunk parameters

- Target size: **~512 tokens**.
- Overlap: **~64 tokens (~12%)**, within-section only (resets at each Item
  boundary).
- Splits are paragraph-aware (never mid-sentence).

Rationale: ~512 keeps every embedding model in play (many cap at 512 tokens) and
is precise for `lookup`; ~12% overlap protects boundary-spanning facts without
bloating the store; within-section overlap avoids cross-topic contamination.
These are tunable defaults — chunk size is the biggest RAG-quality knob; revisit
against the Week 4 retrieval eval.

## Table handling

Financial tables (where exact numbers live) get special treatment:

1. Detect `<table>` elements before the text splitter runs.
2. Keep each table **atomic** — one indivisible chunk, never split or merged
   mid-way.
3. Serialize to **Markdown** (LLM-readable; preserves row/column relationships).
4. Prepend the nearest caption/heading so the chunk is self-describing.

Fallback: a table exceeding the embedding token limit is split on row groups with
headers repeated (edge case, handle in implementation).

## Chunk metadata schema

Each chunk carries:

- `ticker` (e.g. AAPL)
- `fiscal_year` (from CONFORMED PERIOD OF REPORT)
- `filing_type` (10-K)
- `item` / section (e.g. business, risk_factors, mdna, financials)
- `content_type` (narrative | table)
- provenance (source path / accession number)

## Directive-driven retrieval synergy

The directive model already emits structured params (e.g.
`{company: AAPL, year: 2023, metrics: [...]}`). At retrieval time these become
**metadata filters**: restrict to the AAPL-2023-10-K partition, then rank by
semantic similarity. Benefits: higher precision, smaller search space, and the
**corpus-boundary guard** falls out — if a filtered query returns no chunks (e.g.
`company: NFLX`, not in corpus), that empty retrieval signals `decline`. Build the
guard manifest-driven and log the misses (see `docs/future-work.md`).

## Deferred (see docs/future-work.md)

- Adaptive chunk size by section type.
- Corpus expansion (companies / years / filing types), demand-driven via logged
  declines.
- High-value exhibit + XBRL structured-fact ingestion.

## Downstream dependencies

- Embedding model (3.2) must accept >=512-token inputs (our chunk target).
  Choosing a 512-cap model is fine; a longer-context model would let us raise the
  target later.
