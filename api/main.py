# api/main.py
#
# WHAT THIS DEFINES:
#   The FastAPI serving layer for FinDirector. Exposes the pipeline over HTTP so
#   anything (curl, a web page, another service) can send a question and get an
#   answer. This file adds the NON-STREAMING endpoint first (POST /query) — the
#   simplest working API — proving the full wiring before SSE streaming is added.
#
# ENDPOINTS:
#   GET  /health  -> a trivial liveness check ({"status": "ok"}); handy to
#                    confirm the server is up before hitting the real endpoint.
#   POST /query   -> body {"query": "..."} ; runs Orchestrator.answer() and
#                    returns the full Answer as JSON (text, action_code, status,
#                    sources).
#
# INPUT  (POST /query): JSON {"query": "<user question>"}.
# OUTPUT: JSON {"text","action_code","status","sources"}.
#
# STARTUP COST NOTE:
#   The Orchestrator and its dependencies (notably LocalEmbedder, which loads the
#   BGE-M3 model) are built ONCE at import as a module-level singleton and reused
#   for every request — constructing them per-request would be far too slow.

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from scripts.embed.local_embedder import LocalEmbedder
from scripts.generation.anthropic_generator import AnthropicGenerator
from scripts.orchestration.orchestrator import Orchestrator
from scripts.routing.anthropic_router import AnthropicRouter
from scripts.store.pgvector_store import PgVectorStore

# --- Request schema --------------------------------------------------------
# Declares the shape of an incoming /query request. FastAPI parses the JSON body
# into this model and rejects anything without a string "query" field.
class QueryRequest(BaseModel):
    query: str


# --- App + orchestrator singleton ------------------------------------------
app = FastAPI(title="FinDirector API", version="1.0")

# Built once at import, reused across requests (see startup-cost note above).
_orchestrator = Orchestrator(
    router=AnthropicRouter(),
    embedder=LocalEmbedder(),
    store=PgVectorStore(),
    generator=AnthropicGenerator(),
)


# --- Endpoints -------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness check — confirms the server process is up."""
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """Answer one query end-to-end and return the full Answer as JSON."""
    answer = _orchestrator.answer(request.query)
    return asdict(answer)   # Answer dataclass -> plain dict for JSON response