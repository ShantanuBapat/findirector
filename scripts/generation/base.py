"""The Generator interface.

Turns a query + retrieved context into a grounded, sourced answer. Kept behind
an abstract contract so the backend (Anthropic API now, a self-hosted open model
via vLLM later) is a drop-in swap — mirrors the EmbeddingModel/VectorStore
pattern.
"""

from abc import ABC, abstractmethod


class Generator(ABC):
    """Abstract contract: produce an answer from a query and retrieved chunks."""

    @abstractmethod
    def generate(self, query: str, chunks: list[dict]) -> str:
        """Answer `query` grounded in `chunks` (retrieved filing passages).

        Each chunk carries text + metadata (ticker/fiscal_year/section). The
        answer must draw only on the provided chunks and cite their sources.
        Returns the answer as a string.
        """
        ...
