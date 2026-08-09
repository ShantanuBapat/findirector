# scripts/orchestration/answer_types.py
#
# WHAT THIS DEFINES:
#   The Answer dataclass — the single structured shape that the orchestrator's
#   answer() returns, no matter which action code was handled. Every branch
#   (canned no-retrieval replies, the lookup RAG path, the not-yet-built
#   compute/research stubs) produces one of these, so the API layer can handle
#   any result uniformly and expose/log the metadata.
#
# FIELDS:
#   text        - the user-facing response string (always present).
#   action_code - which of the 7 codes was handled (auditability / logging).
#   status      - the KIND of result, independent of the text, so callers can
#                 branch without parsing prose:
#                   "ok"           -> a real answer was produced
#                   "declined"     -> refused (decline code, or corpus-boundary)
#                   "clarify"      -> asked the user for missing info
#                   "unsupported"  -> a valid code whose handler isn't built yet
#                                     (compute / research)
#   sources     - list of citation dicts for lookup hits (ticker/year/section);
#                 empty for everything else.
#
# WHY THIS SHAPE:
#   FinDirector's whole thesis is auditability. Returning a bare string would
#   discard which decision fired and what sources grounded it. The structured
#   Answer keeps the full picture; the API decides what to surface.

from dataclasses import dataclass, field


@dataclass
class Answer:
    """The uniform result of handling one query, whatever the action code."""
    text: str
    action_code: str
    status: str                                   # ok | declined | clarify | unsupported
    sources: list[dict] = field(default_factory=list)