# scripts/orchestration/responses.py
#
# WHAT THIS DEFINES:
#   The canned / static response text for the action codes that don't need
#   retrieval or generation. Kept separate from the orchestrator so the wording
#   can be edited without touching dispatch logic (content vs. control flow).
#
# WHAT'S HERE:
#   SMALLTALK_REPLY  - a single friendly acknowledgement for social turns.
#   META_REPLY       - a static description of what FinDirector can do.
#   DECLINE_REPLIES  - a message per decline reason (keys match the router's
#                      DECLINE_REASONS exactly: investment_advice, prediction,
#                      out_of_scope, personal_financial_advice).
#   UNSUPPORTED_REPLY - honest "not built yet" text for compute / research.
#   decline_text()   - helper that picks the right decline message by reason,
#                      falling back to a generic refusal for an unknown reason.
#
# WHY A FALLBACK IN decline_text():
#   The router should only ever emit the four known reasons, but if params.reason
#   is missing or unexpected, we still return a safe, non-committal refusal rather
#   than a KeyError — the orchestrator must never crash on a malformed directive.

SMALLTALK_REPLY = (
    "Happy to help. Ask me about a company's SEC 10-K filings — for example, "
    "\"What was Apple's R&D spending in fiscal 2023?\""
)

META_REPLY = (
    "FinDirector answers questions about public companies using their SEC 10-K "
    "filings. I can look up specific facts and figures from a company's annual "
    "report and cite the source. I don't give investment advice, make "
    "predictions, or answer questions outside the filings I have."
)

DECLINE_REPLIES = {
    "investment_advice": (
        "I can't provide investment advice or recommendations to buy, sell, or "
        "hold. I can share factual information from a company's SEC filings so "
        "you can do your own analysis."
    ),
    "prediction": (
        "I can't predict future prices, markets, or outcomes. I can report what "
        "a company has disclosed in its SEC filings."
    ),
    "out_of_scope": (
        "That's outside what I do. I answer questions about public companies "
        "using their SEC 10-K filings."
    ),
    "personal_financial_advice": (
        "I can't offer personal financial, tax, or retirement advice. I can "
        "provide factual information from SEC filings for your own research."
    ),
}

_GENERIC_DECLINE = (
    "I can't help with that one, but I'm happy to answer questions about a "
    "company's SEC 10-K filings."
)

UNSUPPORTED_REPLY = (
    "That question needs a capability I don't have yet — {feature}. Right now I "
    "can answer single-company factual lookups from SEC 10-K filings. "
    "Multi-company/multi-year analysis and calculations are on the roadmap."
)


def decline_text(reason: str) -> str:
    """Return the decline message for a reason, or a generic refusal if unknown."""
    return DECLINE_REPLIES.get(reason, _GENERIC_DECLINE)