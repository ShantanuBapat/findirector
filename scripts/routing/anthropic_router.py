# scripts/routing/anthropic_router.py
#
# WHAT THIS DEFINES:
#   AnthropicRouter — the v1 concrete implementation of the DirectiveRouter
#   interface. It classifies a user query into a Directive by calling the
#   Anthropic API with the SAME prompt that generated the training data
#   (prompts/directive_labeler.py). This keeps the live router and the training
#   pipeline identical, and stands in faithfully for the fine-tuned Qwen model
#   until that is served via vLLM in Week 5.
#
# INPUT (to route()):  a raw user query string.
# OUTPUT (from route()): a Directive(action_code, params, reasoning).
#
# HOW IT WORKS:
#   1. Reuse SYSTEM_PROMPT + build_messages() from the shared prompt module.
#   2. Call the Anthropic Messages API (same model/params as the labeler).
#   3. Parse the JSON response defensively (strip code fences, tolerate a bad
#      response by falling back to a safe 'clarify' directive rather than raising)
#      so the orchestrator can always trust route() to return a valid Directive.
#
# WHY 'clarify' AS THE FALLBACK:
#   If the model returns something unparseable, the safest user-facing behavior
#   is to ask for clarification rather than guess a code or crash. It degrades
#   gracefully.

import json
import os

from dotenv import load_dotenv

from prompts.directive_labeler import ACTION_CODES, SYSTEM_PROMPT, build_messages
from scripts.routing.base import Directive, DirectiveRouter

load_dotenv()  # pick up ANTHROPIC_API_KEY from .env

_MODEL = "claude-sonnet-4-5"
_MAX_TOKENS = 512


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ```...``` markdown fence if the model added one.

    The classifier is instructed to return raw JSON, but models sometimes wrap
    it in a fence anyway; stripping it makes json.loads reliable.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline != -1:
        text = text[first_newline + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text


class AnthropicRouter(DirectiveRouter):
    """Classify queries via the Anthropic API, using the training-time prompt."""

    def __init__(self, model: str = _MODEL):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def route(self, query: str) -> Directive:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=build_messages(query),
        )
        raw = response.content[0].text
        cleaned = _strip_code_fences(raw)

        try:
            parsed = json.loads(cleaned)
            action_code = parsed.get("action_code", "")
            params = parsed.get("params", {}) or {}
            reasoning = parsed.get("reasoning", "")
        except json.JSONDecodeError:
            # Unparseable response -> degrade gracefully to a clarify directive.
            return Directive(
                action_code="clarify",
                params={"clarifying_question":
                        "Sorry, could you rephrase your question?"},
                reasoning=f"router parse failure: {raw[:200]}",
            )

        # Guard against an unexpected/invalid code -> also clarify.
        if action_code not in ACTION_CODES:
            return Directive(
                action_code="clarify",
                params={"clarifying_question":
                        "Sorry, could you rephrase your question?"},
                reasoning=f"router returned unknown code {action_code!r}",
            )

        return Directive(action_code=action_code, params=params, reasoning=reasoning)