"""Ingest SEC 10-K filings into tagged, retrievable chunks for the RAG corpus.

Implements the chunking design in docs/design/chunking.md. This module owns the
one step sec-parser does not: unwrapping the full-submission.txt envelope to
extract the primary 10-K document's HTML. Parsing (Items, tables) is delegated
to sec-parser downstream.
"""

import re

# A SEC full submission wraps each document in a <DOCUMENT>...</DOCUMENT> block.
_DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
# Within a block, <TYPE> sits alone on a line (no closing tag), e.g. "<TYPE>10-K".
_TYPE_RE = re.compile(r"<TYPE>([^\r\n]+)")
# The document body lives between <TEXT> and </TEXT>.
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)


def _slice_html(text: str) -> str:
    """Trim the <TEXT> body to just the HTML document.

    iXBRL filings prefix the HTML with an <XBRL> wrapper and an <?xml?>
    declaration; slice from the first <html to the final </html> so the parser
    receives clean HTML. Falls back to the stripped text if no <html> is found.
    """
    lower = text.lower()
    start = lower.find("<html")
    end = lower.rfind("</html>")
    if start == -1 or end == -1:
        return text.strip()
    return text[start:end + len("</html>")]


def extract_primary_document(submission_text: str, doc_type: str = "10-K") -> str:
    """Return the HTML body of the primary document in a full-submission.txt.

    Iterates the submission's <DOCUMENT> blocks and returns the <TEXT> content of
    the first block whose <TYPE> matches `doc_type` exactly — i.e. the actual
    10-K filing, ignoring exhibits (EX-*) and standalone XBRL documents.

    Raises ValueError if no matching document is found, or if the matched
    document has no <TEXT> block.
    """
    for block in _DOCUMENT_RE.finditer(submission_text):
        body = block.group(1)

        type_match = _TYPE_RE.search(body)
        if type_match is None or type_match.group(1).strip() != doc_type:
            continue

        text_match = _TEXT_RE.search(body)
        if text_match is None:
            raise ValueError(f"<TYPE>{doc_type} document has no <TEXT> block")
        return _slice_html(text_match.group(1))

    raise ValueError(f"No <TYPE>{doc_type} document found in submission")


# --------------------------------------------------------------------------- #
# Section identification
# --------------------------------------------------------------------------- #
# 10-K item numbering is fixed by SEC regulation, so we map the item number
# (parsed from the title text) to our own metadata vocabulary. We do NOT use
# sec-parser's section_type: it applies the 10-Q taxonomy and mislabels 10-K
# items (e.g. tags "Item 1. Business" as "Financial Statements").
_ITEM_TO_SECTION = {
    "1": "business",
    "1a": "risk_factors",
    "1b": "unresolved_staff_comments",
    "1c": "cybersecurity",
    "2": "properties",
    "3": "legal_proceedings",
    "4": "mine_safety",
    "5": "market_for_equity",
    "6": "reserved",
    "7": "mdna",
    "7a": "market_risk",
    "8": "financials",
    "9": "accountant_changes",
    "9a": "controls_procedures",
    "9b": "other_information",
    "9c": "foreign_jurisdictions",
    "10": "directors_officers",
    "11": "executive_compensation",
    "12": "security_ownership",
    "13": "related_transactions",
    "14": "accountant_fees",
    "15": "exhibits",
    "16": "form_summary",
}

# Matches an item heading ANCHORED at the start of the text, capturing the item
# number+suffix, e.g. "Item 7." / "Item 1A." The anchor + trailing separator
# prevents matching item-like substrings buried in body sentences. \xa0
# (non-breaking space) is normalized to a space before matching.
_ITEM_RE = re.compile(r"^\s*item\s+(\d+[a-c]?)\s*[.\-\u2013\u2014:]", re.IGNORECASE)


def identify_section(title_text: str) -> str | None:
    """Map a TopSectionTitle's text to our section vocabulary.

    Returns a section name (e.g. "mdna", "financials") for an Item title, or
    None for Part headers ("PART I") and anything that isn't a recognized item.
    """
    normalized = title_text.replace("\xa0", " ")
    match = _ITEM_RE.match(normalized)
    if match is None:
        return None
    return _ITEM_TO_SECTION.get(match.group(1).lower())


# --------------------------------------------------------------------------- #
# Section walk: flat elements -> section-tagged records
# --------------------------------------------------------------------------- #
import sec_parser as sp


def walk_sections(elements: list, metadata: dict) -> list[dict]:
    """Group a flat list of parsed elements into section-tagged records.

    Walks the ordered element list, carrying the current section forward:
    - an element whose text maps to an item (via identify_section) sets the
      current section, flushing the previous section's accumulated narrative;
    - a TableElement is emitted immediately as its own atomic Markdown record;
    - other text (TextElement / SupplementaryText) accumulates into the current
      section's narrative buffer;
    - noise (page numbers/headers, table-of-contents, images, empty) is skipped.

    Each returned record is `metadata` plus:
        {"section": str | None, "content_type": "narrative" | "table", "text": str}

    `section` is None for content before the first recognized item heading.
    """
    records: list[dict] = []
    current_section: str | None = None
    narrative: list[str] = []

    def flush_narrative() -> None:
        """Emit the accumulated narrative for the current section, if any."""
        if not narrative:
            return
        text = "\n\n".join(narrative).strip()
        if text:
            records.append({
                **metadata,
                "section": current_section,
                "content_type": "narrative",
                "text": text,
            })
        narrative.clear()

    for element in elements:
        text = element.text.replace("\xa0", " ").strip()

        # 1. Section boundary? Only TITLE-type elements can be item headings.
        #    (Verified: every 10-K item heading parses as TopSectionTitle or
        #    TitleElement, never TextElement/SupplementaryText. Restricting to
        #    titles + the anchored regex prevents body sentences that mention an
        #    item from being misread as a section switch.)
        if isinstance(element, (sp.TopSectionTitle, sp.TitleElement)):
            section = identify_section(text)
            if section is not None:
                flush_narrative()           # close out the previous section
                current_section = section
                continue

        # 2. Table -> atomic Markdown record, tagged with the current section.
        if isinstance(element, sp.TableElement):
            records.append({
                **metadata,
                "section": current_section,
                "content_type": "table",
                "text": element.table_to_markdown(),
            })
            continue

        # 3. Narrative text -> accumulate.
        if isinstance(element, (sp.TextElement, sp.SupplementaryText)):
            if text:
                narrative.append(text)
            continue

        # 4. Everything else (TitleElement sub-headings, page numbers, TOC,
        #    images, empty) is skipped for narrative purposes.

    flush_narrative()                        # flush the final section
    return records


# --------------------------------------------------------------------------- #
# Size-splitting: section records -> embedding-ready chunks
# --------------------------------------------------------------------------- #
import tiktoken

# Generic encoding for token counting. Chosen to keep the embedding-model choice
# open (counts are close across modern tokenizers). Validate against the chosen
# model's real tokenizer when embeddings are finalized; re-split if any chunk
# overflows its true limit.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Slightly conservative vs the ~512 design target, so generic-token chunks stay
# within a real 512-token embedding limit even if that tokenizer counts higher.
TARGET_TOKENS = 480
OVERLAP_TOKENS = 64
# A sentence-ish split point for paragraphs that exceed the target on their own.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_oversized(text: str) -> list[str]:
    """Split a single over-target block, preferring sentence boundaries.

    Used only when one paragraph alone exceeds TARGET_TOKENS. Falls back to a
    hard token slice if a single 'sentence' is still too big (e.g. a giant
    run-on or delimiter-free text).
    """
    sentences = _SENTENCE_RE.split(text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = _count_tokens(sent)
        if st > TARGET_TOKENS:
            # Flush buffer, then hard-slice this sentence by tokens.
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_tokens = [], 0
            ids = _ENCODING.encode(sent)
            for i in range(0, len(ids), TARGET_TOKENS):
                pieces.append(_ENCODING.decode(ids[i:i + TARGET_TOKENS]))
            continue
        if buf_tokens + st > TARGET_TOKENS and buf:
            pieces.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += st

    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _pack_paragraphs(text: str) -> list[str]:
    """Pack paragraphs into ~TARGET_TOKENS chunks with ~OVERLAP_TOKENS overlap.

    Prefers breaking between paragraphs; only splits within a paragraph when the
    paragraph alone exceeds the target. Overlap carries the tail of one chunk
    into the start of the next so boundary-spanning facts survive.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Pre-split any paragraph that is itself over the target.
    units: list[str] = []
    for p in paragraphs:
        if _count_tokens(p) > TARGET_TOKENS:
            units.extend(_split_oversized(p))
        else:
            units.append(p)

    # Reserve room for the overlap seed so seed + content stays within TARGET.
    content_budget = TARGET_TOKENS - OVERLAP_TOKENS

    def overlap_tail(buffer: list[str]) -> tuple[list[str], int]:
        """Trailing paragraphs of `buffer` totaling up to OVERLAP_TOKENS."""
        tail, tok = [], 0
        for prev in reversed(buffer):
            pt = _count_tokens(prev)
            if tok + pt > OVERLAP_TOKENS:
                break
            tail.insert(0, prev)
            tok += pt
        return tail, tok

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for unit in units:
        ut = _count_tokens(unit)

        # A unit that already exceeds the content budget stands alone: flush any
        # pending buffer, emit the unit as its own chunk, and add NO overlap on
        # top of it (it is already near TARGET). This is the case that otherwise
        # overflowed: a big unit sitting in the buffer + an overlap seed > TARGET.
        if ut > content_budget:
            if buf:
                chunks.append("\n\n".join(buf))
            chunks.append(unit)
            buf, buf_tokens = [], 0
            continue

        if buf_tokens + ut > content_budget and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = overlap_tail(buf)
        buf.append(unit)
        buf_tokens += ut

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def split_records(records: list[dict]) -> list[dict]:
    """Turn section records into embedding-ready chunks.

    Table records pass through atomically. Narrative records are packed into
    ~TARGET_TOKENS chunks with ~OVERLAP_TOKENS overlap. Every output chunk gets
    a `chunk_index` (its position within its source section+type) and a
    `n_tokens` count.
    """
    chunks: list[dict] = []
    for record in records:
        base = {k: v for k, v in record.items() if k != "text"}

        if record["content_type"] == "table":
            text = record["text"]
            chunks.append({**base, "text": text,
                           "chunk_index": 0, "n_tokens": _count_tokens(text)})
            continue

        for i, piece in enumerate(_pack_paragraphs(record["text"])):
            chunks.append({**base, "text": piece,
                           "chunk_index": i, "n_tokens": _count_tokens(piece)})
    return chunks
