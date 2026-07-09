"""
Generate the Colab training notebook (notebooks/train_on_colab.ipynb).

The notebook clones the FinDirector repo, sets up the environment,
and runs the training script on Colab's L4 GPU.

Usage:
    python -m scripts.generate_colab_notebook

Output:
    notebooks/train_on_colab.ipynb
"""

import json
import sys
from pathlib import Path

import nbformat as nbf


OUTPUT_PATH = Path("notebooks/train_on_colab.ipynb")


def markdown_cell(source: str) -> nbf.NotebookNode:
    """Create a markdown cell from source text."""
    return nbf.v4.new_markdown_cell(source.strip())


def code_cell(source: str) -> nbf.NotebookNode:
    """Create a code cell from source text."""
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    """Build the full training notebook."""
    nb = nbf.v4.new_notebook()

    # Add metadata for Colab
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "L4",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }

    # ========================================================================
    # Cell 1: Title and overview
    # ========================================================================
    nb.cells.append(markdown_cell("""
# FinDirector — LoRA Fine-Tuning on Colab

Fine-tune Qwen 2.5 7B Instruct with LoRA adapters on 696 synthetic financial queries.

**Setup:**
- Runtime: L4 GPU (24 GB VRAM)
- Estimated time: 30-90 minutes
- Cost: $0-5 in Colab compute units

**Prerequisites:**
- Colab Pro subscription with L4 GPU access
- Secrets configured: `HF_TOKEN`, `WANDB_API_KEY`
- W&B account with project access

**Pipeline this notebook runs:**
1. Verify GPU allocation
2. Clone the FinDirector repo
3. Install requirements
4. Load secrets from Colab Secrets manager
5. Run `scripts/train_directive_model.py`
6. Save trained adapter to Hugging Face Hub
    """))

    # ========================================================================
    # Cell 2: GPU verification
    # ========================================================================
    nb.cells.append(markdown_cell("## Step 1 — Verify GPU"))
    nb.cells.append(code_cell("""
!nvidia-smi
    """))

    # ========================================================================
    # Cell 3: Clone the repo
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 2 — Clone the FinDirector Repo

Get the latest training script and config from GitHub.
    """))
    nb.cells.append(code_cell("""
# Clone the repo (public HTTPS clone, no auth needed for public repos)
# For private repos, we'd use a GitHub token via Colab Secrets
!git clone https://github.com/ShantanuBapat/findirector.git
%cd findirector
!git log -1 --oneline  # Show the latest commit for reference
    """))

    # ========================================================================
    # Cell 4: Install requirements
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 3 — Install Requirements

Install all training dependencies. Colab has some pre-installed but we
force upgrade to match our pinned versions.
    """))
    nb.cells.append(code_cell("""
# Install all requirements
!pip install -q --upgrade pip
!pip install -q -r requirements.txt
    """))

    # ========================================================================
    # Cell 5: Load secrets
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 4 — Load Secrets from Colab Secrets Manager

Access `HF_TOKEN` and `WANDB_API_KEY` without hardcoding them in the notebook.
    """))
    nb.cells.append(code_cell("""
import os
from google.colab import userdata

# Load secrets into environment variables
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")

# Verify (without printing the actual values)
assert os.environ["HF_TOKEN"].startswith("hf_"), "HF_TOKEN not loaded correctly"
assert len(os.environ["WANDB_API_KEY"]) > 20, "WANDB_API_KEY not loaded correctly"

print("Secrets loaded successfully")
print(f"HF_TOKEN length: {len(os.environ['HF_TOKEN'])}")
print(f"WANDB_API_KEY length: {len(os.environ['WANDB_API_KEY'])}")
    """))

    # ========================================================================
    # Cell 6: Download dataset from HF Hub
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 5 — Download Dataset from HF Hub

Pull the labeled splits from `AlHindi/findirector-splits` (our private dataset repo).

The training script expects splits at `data/synthetic/splits/`.
    """))
    nb.cells.append(code_cell("""
from huggingface_hub import snapshot_download
from pathlib import Path

# Download dataset files to local path expected by training script
splits_dir = Path("data/synthetic/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="AlHindi/findirector-splits",
    repo_type="dataset",
    local_dir=str(splits_dir),
    token=os.environ["HF_TOKEN"],
)

# Verify all 3 files present
for filename in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    path = splits_dir / filename
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"✓ {filename}: {size_kb:.1f} KB")
    else:
        print(f"✗ {filename}: MISSING")
    """))

    # ========================================================================
    # Cell 7: Run the training script
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 6 — Run Training

Execute the training script. Progress logs stream to this notebook's output.

**Expected duration:** 30-90 minutes on L4 GPU.
**W&B dashboard:** https://wandb.ai/{your-username}/findirector-directive-model
    """))
    nb.cells.append(code_cell("""
# Run training as a subprocess so output streams to the notebook
!python -m scripts.train_directive_model
    """))

    # ========================================================================
    # Cell 8: Push adapter to HF Hub
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 7 — Save Trained Adapter to Hugging Face Hub

Push the LoRA adapter (small — ~200 MB) to a private HF Hub model repo
so it persists after this Colab session ends.
    """))
    nb.cells.append(code_cell("""
from huggingface_hub import create_repo, upload_folder
from pathlib import Path

MODEL_REPO_ID = "AlHindi/findirector-directive-lora"
OUTPUT_DIR = Path("outputs/qwen-findirector-lora")

# Create private model repo
create_repo(
    repo_id=MODEL_REPO_ID,
    token=os.environ["HF_TOKEN"],
    repo_type="model",
    private=True,
    exist_ok=True,
)
print(f"Repo ready: {MODEL_REPO_ID}")

# Upload the trained adapter directory
if OUTPUT_DIR.exists():
    upload_folder(
        repo_id=MODEL_REPO_ID,
        folder_path=str(OUTPUT_DIR),
        repo_type="model",
        token=os.environ["HF_TOKEN"],
        commit_message="feat: initial LoRA adapter from Session 2.5 training",
    )
    print(f"Adapter pushed to https://huggingface.co/{MODEL_REPO_ID}")
else:
    print(f"WARNING: {OUTPUT_DIR} does not exist. Training may have failed.")
    """))

    # ========================================================================
    # Cell 9: Summary
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Done!

Trained model available at: **https://huggingface.co/AlHindi/findirector-directive-lora**

**Next steps (Session 2.6):**
- Load the adapter for inference
- Evaluate against held-out test set (156 examples)
- Compare Qwen's predictions to Claude's labels
- Report per-code accuracy and confusion matrix
    """))

    return nb


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    nb = build_notebook()

    with open(OUTPUT_PATH, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook written to: {OUTPUT_PATH}")
    print(f"  Cells: {len(nb.cells)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
