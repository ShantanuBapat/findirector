"""LocalEmbedder: BGE-M3 embeddings via sentence-transformers, run locally."""

from sentence_transformers import SentenceTransformer

from scripts.embed.base import EmbeddingModel

_MODEL_ID = "BAAI/bge-m3"
_DIMENSION = 1024  # BGE-M3's fixed output dimension


def _pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS), else CUDA, else CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbedder(EmbeddingModel):
    """BGE-M3 loaded locally. Model weights download on first construction."""

    def __init__(self, device: str | None = None):
        self.device = device or _pick_device()
        # Loads (and on first run, downloads ~2.3GB) the model onto the device.
        self.model = SentenceTransformer(_MODEL_ID, device=self.device)
        # Cap sequence length at 4096. BGE-M3 supports 8192, but a single
        # ~6.5k-token table padded in a batch OOMs MPS. At 4096 only 7 of 18,742
        # chunks (0.04%, the most extreme tables) are truncated, and peak
        # attention memory stays well within the M4's limits. Improving
        # giant-table embedding (split/average, or GPU) is noted as future work.
        self.model.max_seq_length = 4096

    @property
    def dimension(self) -> int:
        return _DIMENSION

    def embed(
        self, texts: list[str], batch_size: int = 4, show_progress: bool = False
    ) -> list[list[float]]:
        """Embed texts into normalized 1024-dim vectors (order preserved).

        normalize_embeddings=True gives unit-length vectors, which is what
        cosine-distance search (the store's <=> operator) expects.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()
