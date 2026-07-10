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

# Matches "Item 7." / "Item 1A." / "Item 7a" at the start of a title, capturing
# the item number+suffix. \xa0 (non-breaking space) is treated as whitespace.
_ITEM_RE = re.compile(r"item\s*(\d+[a-c]?)\b", re.IGNORECASE)


def identify_section(title_text: str) -> str | None:
    """Map a TopSectionTitle's text to our section vocabulary.

    Returns a section name (e.g. "mdna", "financials") for an Item title, or
    None for Part headers ("PART I") and anything that isn't a recognized item.
    """
    normalized = title_text.replace("\xa0", " ")
    match = _ITEM_RE.search(normalized)
    if match is None:
        return None
    return _ITEM_TO_SECTION.get(match.group(1).lower())
