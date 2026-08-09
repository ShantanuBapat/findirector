# scripts/routing/base.py
#
# WHAT THIS DEFINES:
#   The DirectiveRouter interface — the abstract contract for classifying a user
#   query into a directive: an action code plus extracted parameters. This is the
#   first step of the pipeline (query -> directive), kept behind an abstract base
#   class so the backend is swappable (Anthropic API now, self-hosted vLLM later),
#   exactly like the EmbeddingModel / VectorStore / Generator interfaces.
#
# INPUT (to the method):  a raw user query string.
# OUTPUT (from the method): a Directive — action_code (one of the 7), params (the
#   extracted specifics, e.g. company/year/fact), and reasoning (why this code).
#
# WHY A DATACLASS FOR THE RESULT:
#   The routing result is passed around the orchestrator (to decide handling and
#   to build retrieval filters). A typed Directive object is clearer and safer
#   than a raw dict — callers get named attributes (d.action_code) and mistakes
#   surface early.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Directive:
    """The structured result of classifying one query."""
    action_code: str                              # one of the 7 codes
    params: dict = field(default_factory=dict)    # extracted specifics (company, year, ...)
    reasoning: str = ""                           # the model's stated why (for auditability)


class DirectiveRouter(ABC):
    """Abstract contract: turn a user query into a Directive."""

    @abstractmethod
    def route(self, query: str) -> Directive:
        """Classify `query` into an action code + params.

        Returns a Directive. Implementations must always return a valid
        action_code (falling back to a safe default like 'clarify' if the
        backend response can't be parsed), never raise on a bad response.
        """
        ...