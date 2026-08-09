# scripts/orchestration/orchestrator.py
#
# WHAT THIS DEFINES:
#   The Orchestrator — FinDirector's conductor. Its answer() method takes a raw
#   user query and runs the whole pipeline: route the query into a directive,
#   then dispatch on the action code to produce a uniform Answer. It coordinates
#   the components (router, retriever, generator) but does none of their work
#   itself.
#
# DEPENDENCIES (all injected via __init__, never constructed inside):
#   router     - a DirectiveRouter   (query -> Directive)
#   embedder   - an EmbeddingModel   (used by retrieval)
#   store      - a VectorStore       (used by retrieval)
#   generator  - a Generator         (chunks -> grounded answer)
#   Injection keeps the orchestrator swappable and testable: you can pass fakes
#   in a unit test, or swap the Anthropic router for a vLLM one, without editing
#   this file.
#
# INPUT  (to answer()): a raw user query string.
# OUTPUT (from answer()): an Answer(text, action_code, status, sources).
#
# DISPATCH (by directive.action_code):
#   smalltalk / meta      -> canned/static text, status "ok"
#   clarify               -> the directive's own clarifying_question, status "clarify"
#   decline               -> reason-specific refusal, status "declined"
#   lookup                -> retrieve; hit -> generate (status "ok", with sources);
#                            miss -> corpus-boundary refusal (status "declined")
#   compute / research    -> honest "not built yet" text, status "unsupported"
#   anything else         -> treated as clarify (defensive; router shouldn't emit it)

from scripts.orchestration.answer_types import Answer
from scripts.orchestration.responses import (
    META_REPLY,
    SMALLTALK_REPLY,
    UNSUPPORTED_REPLY,
    decline_text,
)
from scripts.retrieval.retrieve import retrieve
from scripts.routing.base import Directive, DirectiveRouter


class Orchestrator:
    """Coordinates routing -> retrieval -> generation into a single answer()."""

    def __init__(self, router, embedder, store, generator, k: int = 5):
        self.router = router
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.k = k

    def answer(self, query: str) -> Answer:
        """Run the full pipeline for one query and return a uniform Answer."""
        directive: Directive = self.router.route(query)
        code = directive.action_code
        params = directive.params

        if code == "smalltalk":
            return Answer(SMALLTALK_REPLY, code, status="ok")

        if code == "meta":
            return Answer(META_REPLY, code, status="ok")

        if code == "clarify":
            question = params.get(
                "clarifying_question",
                "Could you clarify which company and year you mean?",
            )
            return Answer(question, code, status="clarify")

        if code == "decline":
            return Answer(decline_text(params.get("reason", "")), code,
                          status="declined")

        if code in ("compute", "research"):
            feature = ("calculations on filing data" if code == "compute"
                       else "multi-company / multi-year analysis")
            return Answer(UNSUPPORTED_REPLY.format(feature=feature), code,
                          status="unsupported")

        if code == "lookup":
            return self._handle_lookup(query, params)

        # Defensive: an unexpected code degrades to a clarify-style response.
        return Answer(
            "Could you rephrase your question?", code, status="clarify",
        )

    def _handle_lookup(self, query: str, params: dict) -> Answer:
        """The built RAG path: retrieve, then generate or corpus-boundary decline."""
        result = retrieve(query, params, self.embedder, self.store, k=self.k)

        if result["status"] == "decline":
            company = params.get("company", "that company")
            text = (
                f"I don't have SEC filings for {company} in my corpus, so I "
                f"can't answer that from source. I currently cover a fixed set "
                f"of 20 companies' 10-K filings."
            )
            return Answer(text, "lookup", status="declined")

        chunks = result["chunks"]
        answer_text = self.generator.generate(query, chunks)
        sources = [
            {"ticker": c["ticker"], "fiscal_year": c["fiscal_year"],
             "section": c.get("section")}
            for c in chunks
        ]
        return Answer(answer_text, "lookup", status="ok", sources=sources)