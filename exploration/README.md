# Exploration

Diagnostic probe scripts kept as a record of the empirical verification behind
the ingestion pipeline (`scripts/chunk_filings.py`). These are **not** production
code and are not imported by the pipeline — each one answered a specific question
against a real filing *before* committing to a design choice, following a
"verify against real data, don't guess" discipline.

They are committed deliberately: the SEC-filing parsing landscape is messy, and
these probes document *why* the pipeline is built the way it is — particularly
the decision to detect 10-K sections with our own item-number mapper rather than
`sec-parser`'s section classifier.

Each script is self-contained and run from the repo root, e.g.
`python exploration/test_sections.py`.

## Probes (in the order they were run)

- **`inspect_sections.py`** — dumped the attributes of `sec-parser`'s
  `TopSectionTitle` elements. Finding: its `section_type` applies the **10-Q**
  taxonomy and mislabels 10-K items (e.g. tags "Item 1. Business" with
  title "Financial Statements"). The element `text`, however, is clean and
  correct.

- **`test_sections.py`** — validated `identify_section()` (our item-number →
  section-name mapper) against the parsed section titles. Finding: the mapper
  works, but `Edgar10QParser` only emitted 8 `TopSectionTitle` elements
  (Items 1–6 + the two Part headers) — Items 7 (MD&A) and 8 (Financials) were
  missing from that list.

- **`find_items.py`** — scanned *all* parsed elements (not just
  `TopSectionTitle`) for item headings, showing each one's assigned type.
  Finding: every item heading **is** present; `sec-parser` just types them
  inconsistently (some `TopSectionTitle`, some `TitleElement`). Our mapper
  identifies all of them regardless of type.

- **`check_item7.py`** — confirmed the critical sections MD&A (Item 7),
  Item 5, and Item 12 are present in the element list (they had been hidden in
  the previous probe by a title-length filter, not genuinely absent).

## Conclusion carried into the pipeline

Section boundaries are detected by scanning the flat element list with
`identify_section()` — any element whose text maps to an item is a section
start — rather than trusting `sec-parser`'s `TopSectionTitle` typing or its
`section_type` classification. The `InvalidTopSectionIn10Q` warnings from the
10-Q classifier are suppressed deliberately.
