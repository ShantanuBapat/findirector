"""Evaluate the fine-tuned FinDirector directive model on the held-out test set.

Loads the Qwen base model, attaches the trained LoRA adapter, generates a JSON
answer for each of the 156 test queries (greedy / temperature 0), parses out the
action_code, and reports overall accuracy, per-code precision/recall/F1, a
confusion matrix, and a separate tally of parse failures.

Results are always saved to results/:
  - eval_<date>.json          aggregate metrics
  - eval_<date>_details.jsonl one line per example (query, raw output, verdict)

IMPORTANT — prompt fidelity:
Training (format_for_sft) used NO custom system message. The instruction rode
inside the USER turn as f"{instruction}\n\nQuery: {input}", and Qwen's chat
template auto-inserted its default system prompt. We replicate that here exactly.

Run on a GPU (Colab): python scripts/evaluate_directive_model.py
"""

import json
import os
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from peft import PeftModel
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_ID = "AlHindi/findirector-directive-lora"
DATASET_ID = "AlHindi/findirector-splits"
MAX_NEW_TOKENS = 256
RESULTS_DIR = "results"

# Record schema (from build_splits.py): instruction / input / output / _meta.
INSTRUCTION_FIELD = "instruction"
INPUT_FIELD = "input"
META_FIELD = "_meta"
TRUTH_KEY = "predicted_code"   # nested inside _meta

# Fallback only — every test record already carries its own `instruction`.
# Kept byte-identical to build_splits.py in case a record lacks the field.
INSTRUCTION = (
    "Classify the following financial query into one of 7 action codes "
    "(smalltalk, meta, lookup, compute, research, clarify, decline). "
    "Output JSON with action_code, params, and reasoning."
)

VALID_ACTION_CODES = {
    "smalltalk", "meta", "lookup", "compute",
    "research", "clarify", "decline",
}


# --------------------------------------------------------------------------- #
# 1. Load model + adapter
# --------------------------------------------------------------------------- #
def load_model_with_adapter(
    base_model_id: str,
    adapter_id: str,
    device: torch.device,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the Qwen base model and attach the trained LoRA adapter for inference.

    The adapter is *attached* (PeftModel), not merged, so we evaluate the exact
    artifact we trained — no numerical drift from folding B·A into the base weights.
    """
    start = time.time()

    # 1. Tokenizer — mirror training, but LEFT-pad for batched generation.
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. Base model in the same dtype we trained in (bfloat16).
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
    ).to(device)
    # We disabled the KV cache for training; generation wants it back on.
    base_model.config.use_cache = True

    # 3. Attach the trained LoRA adapter on top of the frozen base.
    model = PeftModel.from_pretrained(base_model, adapter_id)

    # 4. Inference mode — turns off dropout, freezes everything.
    model.eval()

    print(f"Loaded base + adapter on {device} in {time.time() - start:.1f}s")
    return model, tokenizer


# --------------------------------------------------------------------------- #
# 2. Load test set
# --------------------------------------------------------------------------- #
def load_test_set(dataset_id: str, filename: str = "test.jsonl") -> list[dict]:
    """Download the held-out test split from the HF Hub and parse it into records.

    Returns a list of raw dicts; downstream steps pull the instruction+input
    (to rebuild the prompt) and the ground-truth predicted_code out of each one.
    """
    start = time.time()

    local_path = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
    )

    with open(local_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"No records found in {dataset_id}/{filename}")

    print(f"Loaded {len(records)} test records in {time.time() - start:.1f}s")
    print(f"Fields per record: {sorted(records[0].keys())}")
    return records


# --------------------------------------------------------------------------- #
# Prompt + ground-truth reconstruction (must mirror training exactly)
# --------------------------------------------------------------------------- #
def build_user_content(record: dict) -> str:
    """Rebuild the exact USER message used in training.

    format_for_sft used: f"{instruction}\n\nQuery: {input}" with NO system
    message. We read the record's own instruction so it always matches.
    """
    instruction = record.get(INSTRUCTION_FIELD, INSTRUCTION)
    return f"{instruction}\n\nQuery: {record[INPUT_FIELD]}"


def get_truth(record: dict) -> str:
    """Ground-truth code, whether stored top-level or nested under _meta."""
    if TRUTH_KEY in record:
        return record[TRUTH_KEY]
    return record[META_FIELD][TRUTH_KEY]


# --------------------------------------------------------------------------- #
# 3. Generate a prediction
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_prediction(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    """Generate the model's JSON answer for a single query (greedy decoding).

    Builds the SAME prompt shape used in training: a single user turn (no system
    message), then stops before the assistant turn so the model must produce it.
    Qwen's chat template supplies its default system prompt automatically.
    """
    messages = [{"role": "user", "content": user_content}]

    # add_generation_prompt=True appends the assistant header so the model
    # knows to start answering (training data already contained the answer).
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                     # greedy = deterministic ("temperature 0")
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Slice off the prompt tokens; keep only what the model generated.
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# 4. Parse + score
# --------------------------------------------------------------------------- #
def extract_action_code(generated_text: str) -> tuple[str | None, str]:
    """Pull the action_code from a generated JSON string.

    Returns (action_code, status), where status is one of:
    "ok", "empty", "invalid_json", "missing_key", "out_of_taxonomy".
    action_code is None whenever status != "ok".
    """
    if not generated_text.strip():
        return None, "empty"

    # Models sometimes wrap JSON in ```json fences or add stray prose,
    # so grab the first {...} block rather than parsing the whole string.
    match = re.search(r"\{.*\}", generated_text, re.DOTALL)
    if match is None:
        return None, "invalid_json"

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"

    code = parsed.get("action_code")
    if code is None:
        return None, "missing_key"
    if code not in VALID_ACTION_CODES:
        return None, "out_of_taxonomy"

    return code, "ok"


def parse_and_score(
    queries: list[str],
    predictions: list[str],
    ground_truth_codes: list[str],
) -> list[dict]:
    """Score each generated prediction against its ground-truth code.

    Returns one result dict per example:
      {"query", "raw_output", "predicted", "truth", "status", "correct"}
    Parse failures count as incorrect, but their status is retained so we can
    report them separately from misclassifications. The full prompt and raw
    generated text are kept so failures can be inspected without re-running.
    """
    results = []
    for query, text, truth in zip(queries, predictions, ground_truth_codes):
        code, status = extract_action_code(text)
        results.append({
            "query": query,
            "raw_output": text,
            "predicted": code,
            "truth": truth,
            "status": status,
            "correct": status == "ok" and code == truth,
        })
    return results


# --------------------------------------------------------------------------- #
# 5. Compute metrics
# --------------------------------------------------------------------------- #
def compute_metrics(results: list[dict]) -> dict:
    """Aggregate per-example verdicts into report-ready metrics.

    Per-code P/R/F1 and the confusion matrix are computed over *parseable*
    predictions only; overall accuracy and the parse-error tally are computed
    over ALL examples so nothing is hidden.
    """
    total = len(results)
    status_counts = Counter(r["status"] for r in results)
    n_correct = sum(r["correct"] for r in results)

    # Honest headline: parse failures count as wrong.
    overall_accuracy = n_correct / total if total else 0.0

    labels = sorted(VALID_ACTION_CODES)
    valid = [r for r in results if r["status"] == "ok"]
    y_true = [r["truth"] for r in valid]
    y_pred = [r["predicted"] for r in valid]

    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    per_code = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "total": total,
        "n_correct": n_correct,
        "overall_accuracy": overall_accuracy,
        "n_parse_errors": total - status_counts["ok"],
        "status_counts": dict(status_counts),
        "labels": labels,
        "confusion_matrix": matrix,   # rows = true, cols = predicted
        "per_code": per_code,         # precision / recall / f1 / support per code
    }


# --------------------------------------------------------------------------- #
# Reporting + persistence + orchestration
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_report(metrics: dict) -> None:
    """Human-readable dump of the metrics dict."""
    print("\n" + "=" * 52)
    print("FinDirector — directive model evaluation")
    print("=" * 52)
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f} "
          f"({metrics['n_correct']}/{metrics['total']})")
    print(f"Parse errors:     {metrics['n_parse_errors']} -> {metrics['status_counts']}")

    print("\nPer-code (precision / recall / f1 / support):")
    for code in metrics["labels"]:
        m = metrics["per_code"][code]
        print(f"  {code:<10} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1-score']:.3f}  n={int(m['support'])}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    labels = metrics["labels"]
    print(" " * 12 + "".join(f"{l[:6]:>8}" for l in labels))
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  {labels[i][:10]:<10}" + "".join(f"{v:>8}" for v in row))


def save_results(metrics: dict, results: list[dict], out_dir: str = RESULTS_DIR) -> None:
    """Persist aggregate metrics (JSON) and per-example detail (JSONL).

    The details file keeps the full prompt and raw generated text for every
    example, so failures can be inspected later without re-running generation.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    metrics_path = out / f"eval_{stamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    details_path = out / f"eval_{stamp}_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved metrics       -> {metrics_path}")
    print(f"Saved per-example   -> {details_path}")


def main() -> None:
    total_start = time.time()
    login(token=os.environ.get("HF_TOKEN"))   # private model + dataset

    device = get_device()
    model, tokenizer = load_model_with_adapter(BASE_MODEL_ID, ADAPTER_ID, device)
    records = load_test_set(DATASET_ID)

    user_contents = [build_user_content(r) for r in records]
    truths = [get_truth(r) for r in records]

    print(f"\nGenerating on {len(user_contents)} examples (greedy)...")
    gen_start = time.time()
    predictions = []
    for i, user_content in enumerate(user_contents):
        predictions.append(
            generate_prediction(
                model, tokenizer, user_content, device,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(user_contents)} done ({time.time() - gen_start:.0f}s elapsed)")
    print(f"Generation finished in {time.time() - gen_start:.0f}s")

    results = parse_and_score(user_contents, predictions, truths)
    metrics = compute_metrics(results)
    print_report(metrics)
    save_results(metrics, results)

    print(f"\nTotal wall time: {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()