"""The EmbeddingModel interface.

The pipeline embeds text through this abstract contract, not a concrete model,
so the backend (local sentence-transformers now, a scaled GPU/managed service
later) is swappable without touching callers. Mirrors the VectorStore pattern.
"""

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Abstract contract: turn texts into fixed-dimension vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimension of the vectors this model produces (must match the
        vector store's column, e.g. 1024 for BGE-M3)."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text (same order)."""
        ...
