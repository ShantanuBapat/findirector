"""
Generate the Colab evaluation notebook (notebooks/eval_on_colab.ipynb).

The notebook clones the FinDirector repo, sets up the environment, runs the
evaluation script on Colab's L4 GPU against the held-out test set, and prints
any failures (misclassifications + parse errors) for inspection.

Usage:
    python -m scripts.generate_eval_colab_notebook

Output:
    notebooks/eval_on_colab.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf


OUTPUT_PATH = Path("notebooks/eval_on_colab.ipynb")


def markdown_cell(source: str) -> nbf.NotebookNode:
    """Create a markdown cell from source text."""
    return nbf.v4.new_markdown_cell(source.strip())


def code_cell(source: str) -> nbf.NotebookNode:
    """Create a code cell from source text."""
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    """Build the full evaluation notebook."""
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
# FinDirector — Directive Model Evaluation on Colab

Evaluate the fine-tuned Qwen 2.5 7B + LoRA directive model on the 156-example
held-out test set (greedy decoding, temperature 0).

**Setup:**
- Runtime: L4 GPU (24 GB VRAM)
- Estimated time: 5-20 minutes
- Cost: ~$0.15 in Colab compute units

**Prerequisites:**
- Colab Pro subscription with L4 GPU access
- Secrets configured: `HF_TOKEN` (adapter + dataset repos are private)

**Pipeline this notebook runs:**
1. Verify GPU allocation
2. Clone the FinDirector repo
3. Install requirements
4. Load `HF_TOKEN` from Colab Secrets manager
5. Run `scripts/evaluate_directive_model.py` (saves results/ files)
6. Inspect any failures (misclassifications + parse errors)
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

Get the latest evaluation script and config from GitHub.
    """))
    nb.cells.append(code_cell("""
# Clone the repo (public HTTPS clone, no auth needed for public repos)
!git clone https://github.com/ShantanuBapat/findirector.git
%cd findirector
!git log -1 --oneline  # Show the latest commit for reference
    """))

    # ========================================================================
    # Cell 4: Install requirements
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 3 — Install Requirements

Install all dependencies. Colab has some pre-installed but we force upgrade
to match our pinned versions (this also picks up scikit-learn for metrics).

**Note:** Colab ships an old `torchao` that PEFT rejects on adapter load. Upgrade
it here, then **Runtime -> Restart session** before running the eval so the fresh
version is the one imported.
    """))
    nb.cells.append(code_cell("""
# Install all requirements
!pip install -q --upgrade pip
!pip install -q -r requirements.txt
# Fix PEFT's torchao version check (Colab ships < 0.16.0). Restart runtime after.
!pip install -q --upgrade torchao
    """))

    # ========================================================================
    # Cell 5: Load secrets
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 4 — Load Secrets from Colab Secrets Manager

Access `HF_TOKEN` without hardcoding it. Needed because both the adapter
(`AlHindi/findirector-directive-lora`) and the dataset
(`AlHindi/findirector-splits`) are private repos.
    """))
    nb.cells.append(code_cell("""
import os
from google.colab import userdata

# Load secret into environment variable
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# Verify (without printing the actual value)
assert os.environ["HF_TOKEN"].startswith("hf_"), "HF_TOKEN not loaded correctly"

print("Secret loaded successfully")
print(f"HF_TOKEN length: {len(os.environ['HF_TOKEN'])}")
    """))

    # ========================================================================
    # Cell 6: Run the evaluation script
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 5 — Run Evaluation

Execute the evaluation script. It downloads the LoRA adapter + test split from
HF Hub, generates on all 156 examples, prints the report, and saves:
- `results/eval_<date>.json` (metrics)
- `results/eval_<date>_details.jsonl` (per-example, with raw output)

**Expected duration:** 5-20 minutes on L4 GPU.
    """))
    nb.cells.append(code_cell("""
# Run evaluation; report streams here, result files written to results/
!python -m scripts.evaluate_directive_model
    """))

    # ========================================================================
    # Cell 7: Inspect failures
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Step 6 — Inspect Failures

Load the per-example detail file and print every failure (misclassification or
parse error), including the raw generated output. Set `DETAILS` to the dated
file the run just wrote.
    """))
    nb.cells.append(code_cell("""
import glob
import json

# Pick the most recent details file the eval run wrote
detail_files = sorted(glob.glob("results/eval_*_details.jsonl"))
DETAILS = detail_files[-1]
print(f"Reading {DETAILS}\\n" + "=" * 70)

with open(DETAILS) as f:
    rows = [json.loads(line) for line in f if line.strip()]

failures = [r for r in rows if not r["correct"]]
print(f"{len(failures)} failures out of {len(rows)}")

for i, r in enumerate(failures, 1):
    # query stores the full prompt; strip to the user's actual question
    q = r["query"].split("Query:", 1)[-1].strip()
    verdict = r["predicted"] if r["status"] == "ok" else f"PARSE_ERROR({r['status']})"
    print(f"\\n[{i}] true={r['truth']}  ->  got={verdict}")
    print(f"    query: {q}")
    print(f"    raw:   {r['raw_output'][:300]}")
    print("-" * 70)
    """))

    # ========================================================================
    # Cell 8: Summary
    # ========================================================================
    nb.cells.append(markdown_cell("""
## Done!

The report and the failure listing above show the model's classification quality
and the specific queries it got wrong.

**Result files written to `results/`:**
- `eval_<date>.json` — metrics
- `eval_<date>_details.jsonl` — per-example detail (download to commit)

**Next steps (Session 2.6 analysis):**
- Read per-code accuracy and the confusion matrix
- Inspect failures (see the list above) — separate real errors from label noise
- Document findings and commit the results
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
