"""The VectorStore interface.

Retrieval code depends on this abstract contract, not on any concrete store, so
implementations (pgvector now, something else later) are swappable without
touching callers. Concrete stores must implement every abstract method.
"""

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract contract for a vector store: add embedded chunks, search them."""

    @abstractmethod
    def add(self, chunks: list[dict]) -> int:
        """Insert embedded chunks into the store.

        Each chunk is a dict with the chunk metadata/text plus an "embedding"
        key holding its vector (a list[float] of the model's dimension).
        Returns the number of chunks added.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return the k chunks most similar to query_embedding.

        `filters` is an optional dict of exact-match metadata constraints, e.g.
        {"ticker": "AAPL", "fiscal_year": 2023}; only chunks matching all of them
        are considered. Each returned dict includes the chunk fields plus a
        similarity "score". This is where metadata filtering + vector ranking
        combine (the directive-driven retrieval synergy).
        """
        ...
