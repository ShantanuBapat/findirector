# exploration/test_orchestrator.py
#
# WHAT THIS DOES:
#   Exercises the full Orchestrator.answer() pipeline end to end — raw query in,
#   Answer out — across the handling categories, with all real components wired
#   in (Anthropic router, BGE-M3 embedder, pgvector store, Anthropic generator).
#   This is the first time routing -> retrieval -> generation runs as a single
#   call rather than hand-chained pieces.
#
# INPUT:  none (queries hardcoded). Requires a running pgvector DB (docker compose
#   up) and ANTHROPIC_API_KEY in .env.
# OUTPUT: for each query, prints the action_code, status, any sources, and the
#   answer text — so we can confirm each branch behaves correctly.
#
# NOTE: this makes real API calls (router + generation) and hits the DB, so it's
#   slower than the earlier syntax checks — a few seconds per query.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.embed.local_embedder import LocalEmbedder
from scripts.generation.anthropic_generator import AnthropicGenerator
from scripts.orchestration.orchestrator import Orchestrator
from scripts.routing.anthropic_router import AnthropicRouter
from scripts.store.pgvector_store import PgVectorStore

# Wire up the real pipeline (dependency injection).
orch = Orchestrator(
    router=AnthropicRouter(),
    embedder=LocalEmbedder(),
    store=PgVectorStore(),
    generator=AnthropicGenerator(),
)

# One query per branch, to see the whole taxonomy handled by a single call.
queries = [
    "What was Apple's R&D spending in fiscal 2023?",       # lookup (hit) -> real answer
    "What were Netflix's revenues in 2023?",               # lookup (miss) -> corpus-boundary decline
    "Compare Apple and Microsoft's 2023 margins.",         # research -> unsupported
    "Should I buy Tesla stock?",                           # decline -> refusal
    "What were earnings last quarter?",                    # clarify -> clarifying question
    "Thanks!",                                             # smalltalk -> canned
    "What can you do?",                                    # meta -> static help
]

for q in queries:
    a = orch.answer(q)
    print(f"\n{'='*70}\nQUERY: {q}")
    print(f"  code={a.action_code}  status={a.status}  sources={len(a.sources)}")
    print(f"  answer: {a.text[:220]}")