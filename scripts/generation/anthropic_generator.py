"""AnthropicGenerator: grounded answer generation via the Anthropic API."""

import os

from dotenv import load_dotenv

from scripts.generation.base import Generator

load_dotenv()  # pick up ANTHROPIC_API_KEY from .env

_MODEL = "claude-sonnet-4-5"
_MAX_TOKENS = 1024

_SYSTEM = (
    "You answer questions about public companies using ONLY the provided SEC "
    "10-K filing excerpts. Rules:\n"
    "- Use only the excerpts below; do not use outside knowledge.\n"
    "- If the excerpts do not contain the answer, say so plainly rather than "
    "guessing.\n"
    "- Cite the source of each fact inline as (TICKER FY, section).\n"
    "- Be concise and precise with figures.\n"
    "- Do not give investment advice, predictions, or recommendations."
)


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['ticker']} FY{c['fiscal_year']} — {c.get('section')}"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


class AnthropicGenerator(Generator):
    """Generate grounded answers with a Claude model via the Anthropic API."""

    def __init__(self, model: str = _MODEL):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def generate(self, query: str, chunks: list[dict]) -> str:
        context = _format_context(chunks)
        user_msg = (
            f"Question: {query}\n\n"
            f"SEC 10-K excerpts:\n{context}\n\n"
            f"Answer using only these excerpts, citing sources inline."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
