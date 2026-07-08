"""
Push train/val/test JSONL splits to a private Hugging Face Hub dataset.

Reads HF_TOKEN from .env. Creates or updates a private dataset repo
under the current user's namespace. Uploads three JSONL files plus
an auto-generated README describing the dataset.

Usage:
    python -m scripts.push_splits_to_hf

The resulting dataset can be loaded on Colab (or anywhere) with:
    from datasets import load_dataset
    ds = load_dataset("AlHindi/findirector-splits")

Cost: $0 (free tier).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file


# Configuration
HF_USERNAME = "AlHindi"
DATASET_NAME = "findirector-splits"
DATASET_REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"
PRIVATE = True  # keep private; can be flipped to False for public sharing later

SPLITS_DIR = Path("data/synthetic/splits")
FILES_TO_UPLOAD = ["train.jsonl", "val.jsonl", "test.jsonl"]


DATASET_README = """\
---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- financial
- sec-filings
- directive-classification
- lora-fine-tuning
size_categories:
- n<1K
---

# FinDirector — Directive Classification Splits

Training data for fine-tuning a directive model (Qwen 2.5 7B + LoRA) that
classifies financial queries into 7 action codes: smalltalk, meta, lookup,
compute, research, clarify, decline.

## Dataset structure

Three splits in instruction-following format:

| Split | Records | Purpose |
|-------|---------|---------|
| train | 696 | Fine-tuning training set |
| val   | 146 | Per-epoch loss monitoring, early stopping |
| test  | 156 | Held-out evaluation set |

Each record:

```json
{
  "instruction": "Classify the following financial query into one of 7 action codes...",
  "input": "What was Apple's R&D in 2023?",
  "output": "{\\"action_code\\": \\"lookup\\", \\"params\\": {...}, \\"reasoning\\": \\"...\\"}",
  "_meta": {
    "intended_code": "lookup",
    "predicted_code": "lookup",
    "agrees": true
  }
}
```

## Pipeline provenance

Generated via a two-prompt distillation pipeline:

1. **Query generator** (Claude Sonnet 4.5) — produced 1,400 synthetic queries
   across the 7-code taxonomy using per-code specialized prompts
2. **Deduplication** — normalized 1,400 → 998 unique queries
3. **Labeler** (Claude Sonnet 4.5) — classified each query with structured
   output (action_code + params + reasoning)
4. **Cross-validation** — 96.8% agreement between generator intent and
   labeler verdict (strong taxonomy consistency signal)
5. **Surgical overrides** — 3 corpus-boundary corrections applied
6. **Stratified split** — 70/15/15 with fixed seed (42), stratified by
   predicted_code, all sanity checks passed

## Per-code distribution

| Code | Train | Val | Test | Total |
|------|-------|-----|------|-------|
| smalltalk | 43 | 9 | 10 | 62 |
| meta | 97 | 20 | 22 | 139 |
| lookup | 135 | 28 | 30 | 193 |
| compute | 94 | 20 | 21 | 135 |
| research | 103 | 22 | 23 | 148 |
| clarify | 91 | 19 | 20 | 130 |
| decline | 133 | 28 | 29 | 190 |

## Corpus scope

Queries reference companies from a curated 20-ticker set:
- Big Tech: AAPL, MSFT, GOOGL, AMZN, META, NVDA
- Finance: JPM, BAC, BRK-B, V
- Healthcare: JNJ, UNH, PFE
- Consumer: WMT, KO, PG
- Energy: XOM, CVX
- Industrial/Auto: TSLA, BA

Time range: 2022-2024 (10-K annual filings only).

## Reproducibility

Full pipeline code at:
https://github.com/ShantanuBapat/findirector

## License

MIT.
"""


def main() -> int:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Check .env.")
        return 1

    if not SPLITS_DIR.exists():
        print(f"ERROR: {SPLITS_DIR} not found. Run scripts/build_splits.py first.")
        return 1

    # Verify all files exist locally
    missing = []
    for filename in FILES_TO_UPLOAD:
        if not (SPLITS_DIR / filename).exists():
            missing.append(filename)
    if missing:
        print(f"ERROR: Missing split files: {missing}")
        return 1

    api = HfApi(token=hf_token)

    # Create or verify the dataset repo exists
    print(f"Creating (or confirming) dataset repo: {DATASET_REPO_ID}")
    try:
        create_repo(
            repo_id=DATASET_REPO_ID,
            token=hf_token,
            repo_type="dataset",
            private=PRIVATE,
            exist_ok=True,
        )
        print(f"  Repo ready. Private: {PRIVATE}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        return 1

    # Upload the README first
    print("\nUploading README.md...")
    try:
        upload_file(
            path_or_fileobj=DATASET_README.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=hf_token,
            commit_message="docs: update dataset README",
        )
        print("  README uploaded.")
    except Exception as e:
        print(f"ERROR uploading README: {e}")
        return 1

    # Upload each split file
    print("\nUploading split files...")
    for filename in FILES_TO_UPLOAD:
        local_path = SPLITS_DIR / filename
        size_kb = local_path.stat().st_size / 1024

        print(f"  {filename} ({size_kb:.1f} KB)...", end=" ", flush=True)
        try:
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                token=hf_token,
                commit_message=f"data: upload {filename}",
            )
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    # Print the dataset URL
    print(f"\n{'=' * 60}")
    print("SUCCESS")
    print(f"{'=' * 60}")
    print(f"Dataset URL: https://huggingface.co/datasets/{DATASET_REPO_ID}")
    print(f"\nTo load in Colab or elsewhere:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('{DATASET_REPO_ID}', token=hf_token)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
