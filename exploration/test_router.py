# exploration/test_router.py
#
# WHAT THIS DOES:
#   Smoke-tests the AnthropicRouter end to end against a few real queries — one
#   per major handling category — to confirm (a) the API key works, (b) the
#   prompt classifies correctly, and (c) route() returns a valid Directive with
#   the expected action_code and params.
#
# INPUT:  none (queries are hardcoded below). Requires ANTHROPIC_API_KEY in .env.
# OUTPUT: for each query, prints the returned action_code, params, and a short
#   slice of the reasoning — so we can eyeball that routing is correct.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.routing.anthropic_router import AnthropicRouter

router = AnthropicRouter()

# One query per handling category, to see routing across the taxonomy.
queries = [
    "What was Apple's R&D spending in fiscal 2023?",      # expect: lookup
    "Compare Apple and Microsoft's 2023 operating margins.",  # expect: research
    "Should I buy Tesla stock?",                          # expect: decline
    "What were the earnings last quarter?",               # expect: clarify (no company)
    "Thanks, that was helpful!",                          # expect: smalltalk
]

for q in queries:
    d = router.route(q)
    print(f"\nQUERY: {q}")
    print(f"  action_code: {d.action_code}")
    print(f"  params:      {d.params}")
    print(f"  reasoning:   {d.reasoning[:90]}...")