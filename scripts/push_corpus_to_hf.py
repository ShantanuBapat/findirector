"""Push the chunk corpus to a HuggingFace dataset repo.

The corpus (data/chunks/corpus.jsonl) is a large derived artifact, gitignored
locally, so HF is its canonical shareable home — this is how Colab (GPU
embedding) and future runs obtain it. Requires HF_TOKEN (write) in the env.
"""

from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()  # pick up HF_TOKEN from .env

REPO_ID = "AlHindi/findirector-corpus"
CORPUS_PATH = "data/chunks/corpus.jsonl"


def main() -> None:
    api = HfApi()
    corpus = Path(CORPUS_PATH)
    if not corpus.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} not found — run chunking first")

    size_mb = corpus.stat().st_size / 1e6
    print(f"uploading {CORPUS_PATH} ({size_mb:.1f} MB) -> {REPO_ID}")

    # Create the dataset repo if it doesn't exist (idempotent).
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    # Upload the corpus file.
    api.upload_file(
        path_or_fileobj=str(corpus),
        path_in_repo="corpus.jsonl",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
