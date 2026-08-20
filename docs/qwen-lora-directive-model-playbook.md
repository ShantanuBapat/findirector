# Training the Qwen 2.5 7B + LoRA "Directive Model"

**Reconstructed end-to-end from `github.com/ShantanuBapat/findirector`** — synthetic data generation through trained adapter and held-out evaluation.

Result achieved: **94.87% (148/156)** on a held-out test set, with an adapter of ~40M trainable params (~0.5% of the 7B base).

Timeline in the repo: **2026-06-25 → 2026-07-09** (roughly two weeks of sessions, ~4 days of actual pipeline work).

---

## 0. What the model actually does

Not a chatbot. A **classifier that emits structured JSON**. Input is one user query; output is:

```json
{
  "action_code": "lookup",
  "params": {"company": "AAPL", "year": 2023, "fact_requested": "R&D spending"},
  "reasoning": "Single company, single year, single fact. Standard retrieval from one filing."
}
```

Seven mutually-exclusive action codes: `smalltalk`, `meta`, `lookup`, `compute`, `research`, `clarify`, `decline`.

The whole training approach is **distillation**: Claude Sonnet 4.5 is the teacher (it generates *and* labels the data), Qwen 2.5 7B + LoRA is the student. There was never a human-labeled corpus.

---

## Part I — Generating the training data

### Step 1 — Define the taxonomy as code, not prose

**File:** `prompts/directive_labeler.py`
**Commit:** `feat: design directive-classification labeling prompt with 100% eval accuracy` (2026-06-29)

The seven codes and the four decline sub-reasons were made a single source of truth:

```python
ACTION_CODES = ("smalltalk", "meta", "lookup", "compute", "research", "clarify", "decline")
DECLINE_REASONS = ("investment_advice", "prediction", "out_of_scope", "personal_financial_advice")
```

Every downstream script imports `ACTION_CODES` from here — generation, dedup, labeling, splits, eval. Nothing re-declares the taxonomy.

The decision boundary was defined as **retrieval pattern, not topic complexity**:

```
single doc + single fact        → lookup
single doc + arithmetic         → compute
multi doc OR multi time period  → research
```

### Step 2 — Build the labeling prompt (the teacher)

Same file. A deliberately verbose ~400-line `SYSTEM_PROMPT` with six components:

1. **Role framing** — "You think like a router: your job is to decide WHAT KIND of work the query requires, not to answer it."
2. **Task definition** — read query → pick one code → extract params → emit JSON. Plus explicit *will NOT* list (don't answer, don't multi-label, don't add commentary).
3. **Per-code definitions** — one block per code with routing consequence.
4. **Six ordered tie-break rules** — first rule that fires wins:
   - Rule 1: **decline takes precedence** (safety routing overrides retrieval routing)
   - Rule 2: clarify on missing required context — *do not guess defaults*
   - Rule 3: meta for questions about the system
   - Rule 4: the retrieval-pattern decision tree
   - Rule 5: **topic complexity is not a signal**
   - Rule 6: social content is smalltalk even if financial-adjacent
5. **Ten few-shot examples** — one per code plus the hard edge cases (conceptually-deep-but-still-lookup, decline-disguised-as-lookup).
6. **Output contract** — exact JSON shape, per-code `params` schema, validation requirements (no trailing commas, `null` not `""`, tickers not company names, `reasoning` ≥ one sentence).

This prompt was validated on a small hand-built eval set before generating anything at scale — the commit message records **100% accuracy** on it. That gate mattered: every label downstream inherits this prompt's quality.

### Step 3 — Build the query generator (three-layer prompt)

**File:** `prompts/synthetic_query_generator.py`
**Commit:** `feat: synthetic query generation pipeline` (2026-06-29)

Deliberately kept **separate** from the labeler so the generator can't leak the label into the query. The generator produces *only* query strings; classification happens in a separate downstream call.

Architecture — "Option C, hybrid":

- **Layer 1: `BASE_INSTRUCTIONS`** — shared scaffold, identical for all seven codes:
  - **Corpus constraints** — the exact 20 tickers, grouped by sector (Big Tech, Finance, Healthcare, Consumer, Energy, Industrial/Auto), years 2022–2024 only. Explicit negatives: *"Do NOT generate queries about Netflix, Spotify, Disney."*
  - **Diversity requirements** — forced variation across company, time reference (specific / relative / fiscal notation), formality (casual / neutral / formal), question structure, and length (5–10 / 15–25 / 30+ words).
  - **Anti-patterns** — no multi-part queries, no repeats within a batch, no meta-commentary, no queries that *are* labels.
  - **Output format** — bare JSON array of N strings, no fences, no prose.
- **Layer 2: `PER_CODE_GUIDANCE`** — a dict keyed by code. Each entry has the same four sections:
  - *Defining traits*
  - *Sub-types to include in each batch* (5–7 of them — e.g. for `lookup`: specific numbers, definitional, disclosure, section-specific, mention/inclusion, quote-extraction)
  - *Edge cases to specifically include* — this is where the value is. E.g. for `smalltalk`: "Smalltalk that MENTIONS a company but asks nothing (`Apple's been busy lately`) — tests that the directive model doesn't auto-route on entity presence."
  - *Specifically AVOID* — the neighbouring codes, with the reason. E.g. `lookup` avoid-list explicitly redirects multi-company → research, arithmetic → compute, investment angle → decline.
  - For `decline` only, an explicit **sub-reason distribution**: investment_advice 35%, prediction 25%, out_of_scope 25%, personal_financial_advice 15%.
- **Layer 3: `build_generation_prompt(code)`** — concatenates Layer 1 + Layer 2.

The negative space (avoid-lists + edge cases) is what produced clean decision boundaries. The `compute` guidance, for instance, explains that within-filing year-over-year is `compute` (one 10-K shows current + prior year) while across-filing multi-year is `research`. That single distinction was later learned perfectly by the student.

### Step 4 — Generate at scale

**File:** `scripts/generate_synthetic_queries.py`

```
TARGET_PER_CODE  = 200
QUERIES_PER_CALL = 10
MODEL            = claude-sonnet-4-5, max_tokens=4096
OUTPUT           = data/synthetic/raw/{code}.jsonl
```

Design points:

- **Batches of 10, not 200.** Small batches keep in-batch diversity enforceable and keep any single failure cheap.
- **Append-per-batch to JSONL.** Each batch is written immediately, so a crash at query 170 doesn't lose the first 169.
- Truncate-if-exists guard so a re-run doesn't silently append to a partial file.
- One retry with a short sleep on batch failure; then skip and continue.
- Defensive parse: `_strip_code_fences()` handles the model wrapping JSON in ```` ```json ````, then `json.loads`, then type-check that it's a list of strings.
- Token usage tracked per call → live cost accounting (`input × $3/M + output × $15/M`).
- Prints a per-code duplicate-rate check at the end.

Each output line:

```json
{"code": "lookup", "query": "What was Apple's R&D in 2023?", "batch_id": 3}
```

**Output: 1,400 raw queries** (200 × 7). Projected cost ≈ $1.50.

### Step 5 — Deduplicate

**File:** `scripts/dedup_synthetic_queries.py` → `data/synthetic/dedup/{code}.jsonl`

- Normalization for comparison: `" ".join(query.lower().split())` — case- and whitespace-insensitive.
- **Original casing/spacing is preserved in the output** — only the comparison key is normalized.
- **Within-code only.** Cross-code duplicates are deliberately kept: a `"Hi"` generated for smalltalk and one generated for clarify could legitimately coexist.

**1,400 → 998 unique (~29% duplicate rate).** Deduping *before* labeling saved ~400 labeling calls (~$4.40).

### Step 6 — Label the 998

**File:** `scripts/label_synthetic_queries.py`

```
MODEL       = claude-sonnet-4-5, max_tokens=512
NUM_WORKERS = 5 (ThreadPoolExecutor)
INPUT       = data/synthetic/dedup/{code}.jsonl
OUTPUT      = data/synthetic/labeled/{code}.jsonl + data/synthetic/disagreements.jsonl
```

The critical design choice: **the generator's intent and the labeler's verdict are recorded separately.**

```json
{
  "query": "...",
  "intended_code": "lookup",     // what the generator was asked to produce
  "predicted_code": "compute",   // what the independent labeler decided
  "params": {...},
  "reasoning": "...",
  "agrees": false
}
```

Disagreements are streamed to a separate file as they happen. This gives a free **quality signal on the taxonomy itself** — a code with low agreement means the boundary is fuzzy, not that the model is bad. The run prints per-code agreement %, total disagreement rate, and a top-10 confusion table (`intended → predicted`).

Engineering details: thread-safe append under a `Lock`; one retry with 2s backoff on API failure; parse failures recorded as `action_code: "PARSE_ERROR"` rather than crashing the run; progress + ETA every 25 queries.

Projected cost ≈ $11 for 998 queries (~$0.011 each).

### Step 7 — Surgical manual overrides

**File:** `scripts/apply_label_overrides.py`

Reviewing the disagreements exposed a **known blind spot**: the labeler prompt describes the taxonomy but not the *corpus boundary*. It doesn't know only 20 tickers and only 10-Ks exist. So it labeled "What was Netflix's revenue in fiscal 2023?" as a perfectly reasonable `lookup` — when the right answer for this system is `decline / out_of_scope`.

Exactly **three** overrides were applied:

| Query | → | Reason |
|---|---|---|
| "Tell me about Netflix's streaming subscriber growth in 2023" | decline / out_of_scope | Netflix not in the 20-ticker corpus |
| "What was Netflix's revenue in fiscal 2023?" | decline / out_of_scope | same |
| "What does Pfizer's 10-Q say about their pipeline?" | decline / out_of_scope | 10-Q not in corpus (10-K only) |

Implementation notes worth keeping:

- Matched on **normalized query text, not index** — indices are unstable across re-runs.
- The override rewrites `predicted_code`, `params`, prepends `[OVERRIDE] {justification}` to `reasoning` (keeping the original labeler reasoning truncated), sets `agrees` correctly, and stamps `override_applied: true`.
- `disagreements.jsonl` is **regenerated from scratch** afterwards so it stays consistent.
- Warns loudly if an override target isn't found.

### Step 8 — Stratified splits

**File:** `scripts/build_splits.py` → `data/synthetic/splits/{train,val,test}.jsonl`
**Commit:** `feat: stratified train/val/test split` (2026-06-30)

```
70 / 15 / 15, stratified by predicted_code, seed=42
998 → train 696 · val 146 · test 156
```

Label source is **`predicted_code`** (the labeler's verdict + the 3 overrides), *not* `intended_code`. The generator's intent is kept only as metadata.

Converted to instruction-following format:

```json
{
  "instruction": "Classify the following financial query into one of 7 action codes (smalltalk, meta, lookup, compute, research, clarify, decline). Output JSON with action_code, params, and reasoning.",
  "input": "<the query text>",
  "output": "<JSON string of {action_code, params, reasoning}>",
  "_meta": {"intended_code": "...", "predicted_code": "...", "agrees": true}
}
```

Four sanity checks run automatically and are the reason this step is a script and not a notebook cell:

1. **Total preserved** — 998 in, 998 out (test takes the remainder so rounding never drops records).
2. **No leakage** — no query text appears in more than one split.
3. **Stratification holds** — each code's per-split fraction within 0.05 of target.
4. **Min-per-class** — every code appears at least once in every split.

### Step 9 — Publish the dataset

**File:** `scripts/push_splits_to_hf.py` → `AlHindi/findirector-splits` (private HF dataset repo)

Uploads the three JSONL files plus an auto-generated README with YAML frontmatter (license, task_categories, tags, size_categories). This exists so **Colab pulls data from a URL, not from a local disk** — the training environment becomes stateless and reproducible.

---

## Part II — Training the LoRA adapter

### Step 10 — Lock the hyperparameters in a config module

**File:** `scripts/training_config.py`
**Commit:** `feat: LoRA fine-tuning script for Qwen 2.5 7B directive model` (2026-07-08)

Every hyperparameter lives here as a typed module constant with a comment explaining the decision. The training script imports; it declares nothing.

**Model**
| | |
|---|---|
| `BASE_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` |
| `BASE_MODEL_DTYPE` | `bfloat16` — same numerical range as fp32, half the memory |

**LoRA**
| | | Rationale |
|---|---|---|
| `LORA_R` | `16` | sweet spot for classification-style tasks |
| `LORA_ALPHA` | `32` | 2:1 alpha:rank |
| `LORA_DROPOUT` | `0.05` | |
| `LORA_TARGET_MODULES` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | **all linear layers**, not just attention |
| `bias` | `"none"` | |
| `task_type` | `CAUSAL_LM` | |

→ ≈ **40M trainable params, ~0.5% of the model** (printed at runtime by `configure_lora`).

**Optimization**
| | |
|---|---|
| `LEARNING_RATE` | `3e-4` — LoRA tolerates far higher LR than full fine-tuning |
| `WEIGHT_DECAY` | `0.01` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `4` |
| `GRADIENT_ACCUMULATION_STEPS` | `8` |
| **Effective batch size** | **32** |
| `NUM_TRAIN_EPOCHS` | `3` |
| `WARMUP_RATIO` | `0.1` |
| `LR_SCHEDULER_TYPE` | `cosine` |
| `OPTIMIZER` | `adamw_torch` |
| `MAX_SEQ_LENGTH` | `512` — data tops out ~300 tokens, so this is headroom |
| `TOKENIZER_PADDING_SIDE` | `right` (training) |
| `TRAINING_SEED` | `42` |

At 696 examples / effective batch 32 → ~21 optimizer steps per epoch, ~63 total.

**Logging & checkpointing**
| | |
|---|---|
| `REPORT_TO` | `wandb`, project `findirector-directive-model` |
| `LOGGING_STEPS` | `5` |
| `EVALUATION_STRATEGY` / `SAVE_STRATEGY` | `epoch` |
| `SAVE_TOTAL_LIMIT` | `2` |
| `LOAD_BEST_MODEL_AT_END` | `True`, on `eval_loss`, lower-is-better |

### Step 11 — The training script

**File:** `scripts/train_directive_model.py`

Six clean stages, each a separate function:

1. **`load_tokenizer()`** — Qwen has no `pad_token`; set `pad_token = eos_token`. `padding_side = "right"` for training.
2. **`load_base_model()`** — `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16, device_map=<cuda|mps>)`. Then **`model.config.use_cache = False`** (KV cache must be off during training).
3. **`configure_lora()`** — build `LoraConfig`, `get_peft_model()`, print trainable / total / percentage.
4. **`load_jsonl_as_dataset()`** — JSONL → HF `Dataset` via `Dataset.from_list`.
5. **`format_for_sft()`** — the prompt-shape decision, and the one to be most careful about:

```python
user_content = f"{record['instruction']}\n\nQuery: {record['input']}"
return {"messages": [
    {"role": "user",      "content": user_content},
    {"role": "assistant", "content": record["output"]},
]}
```

   **No custom system message.** The instruction rides inside the *user* turn; Qwen's chat template auto-inserts its own default system prompt. `SFTTrainer` applies the chat template and masks loss to the assistant turn only. This exact shape has to be replicated at inference or accuracy drops — the eval script has a comment block about it for that reason.

6. **`SFTConfig` + `SFTTrainer`** — `bf16=True`, `remove_unused_columns=False` (so `_meta` survives), `processing_class=tokenizer`, `max_length=512`.

Then `trainer.train()` → `trainer.save_model()` → `tokenizer.save_pretrained()` into `outputs/qwen-findirector-lora/`.

A `get_device()` helper picks CUDA > MPS > CPU, so the same script smoke-tests on an M4 Mac and runs for real on a GPU.

### Step 12 — Run it on Colab

**Files:** `scripts/generate_colab_notebook.py` → `notebooks/train_on_colab.ipynb`
**Commit:** `feat: Colab notebook for LoRA training on L4 GPU` (2026-07-09)

The notebook is **generated programmatically with `nbformat`**, not hand-edited — so it stays in version control as reviewable Python and can't drift from the training script.

Runtime: **L4 GPU, 24 GB VRAM**. Seven cells:

1. `!nvidia-smi` — verify GPU allocation before spending anything.
2. `git clone` the repo, `%cd findirector`, `git log -1 --oneline` to stamp the exact commit into the run log.
3. `pip install -r requirements.txt`, then `pip install --upgrade torchao` (PEFT's version check fails against Colab's shipped torchao; restart runtime after).
4. Load `HF_TOKEN` + `WANDB_API_KEY` from **Colab Secrets** (`google.colab.userdata`) into env vars — asserted for shape, never printed.
5. `snapshot_download("AlHindi/findirector-splits", repo_type="dataset")` into `data/synthetic/splits/`, then verify all three files exist with sizes.
6. `!python -m scripts.train_directive_model` — logs stream to the notebook, metrics to W&B.
7. `create_repo` + `upload_folder` → push the adapter to `AlHindi/findirector-directive-lora` (private, ~200 MB).

**Actual run: 30–90 minutes, $0–5 in Colab compute units.**

Step 7 is the one that matters most operationally — Colab sessions die and take the local disk with them. The adapter is pushed to the Hub in the same notebook run that produced it.

---

## Part III — Evaluating it

### Step 13 — Real autoregressive evaluation

**File:** `scripts/evaluate_directive_model.py` (+ `notebooks/eval_on_colab.ipynb`)
**Commit:** `results: directive model eval, 94.87% on test set` (2026-07-09)

Not loss-based. Actual generation, parsed and scored.

```
BASE_MODEL_ID  = Qwen/Qwen2.5-7B-Instruct
ADAPTER_ID     = AlHindi/findirector-directive-lora
DATASET_ID     = AlHindi/findirector-splits (test.jsonl, 156 examples)
Decoding       = greedy (do_sample=False) — deterministic, "temperature 0"
MAX_NEW_TOKENS = 256
Hardware       = Colab L4
```

The four things this script gets right and that are easy to get wrong:

1. **Adapter is attached, not merged.** `PeftModel.from_pretrained(base, adapter_id)` — evaluate the exact artifact you trained, with no numerical drift from folding B·A into base weights.
2. **`use_cache` flipped back on** for generation (training turned it off), and `model.eval()`.
3. **`padding_side = "left"`** for generation, versus `"right"` for training.
4. **Prompt fidelity.** `build_user_content()` reconstructs `f"{instruction}\n\nQuery: {input}"` from the record's *own* instruction field, with no system message, then `apply_chat_template(..., add_generation_prompt=True)`. Byte-identical to training.

Parsing is defensive: regex the first `{.*}` block out of the output (models add fences and stray prose), `json.loads`, then check `action_code` is in the taxonomy. Status is one of `ok | empty | invalid_json | missing_key | out_of_taxonomy`.

**Scoring is honest**: parse failures count as *wrong* in the headline accuracy. Per-code P/R/F1 and the confusion matrix are computed over parseable outputs only, and the parse-error tally is reported separately so nothing is hidden.

Both aggregate metrics (`results/eval_<date>.json`) and **per-example detail** (`results/eval_<date>_details.jsonl` — query, raw output, predicted, truth, status, correct) are always saved. That second file is what made the failure analysis possible without re-running an 895-second generation pass.

### Results

**Overall: 94.87% (148/156).** 2 parse errors (1.3%). Among 154 valid outputs, ~96%.

| Code | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| clarify | 1.000 | 0.842 | 0.914 | 19 |
| compute | 1.000 | 1.000 | **1.000** | 21 |
| decline | 1.000 | 0.931 | 0.964 | 29 |
| lookup | 0.909 | 1.000 | 0.952 | 30 |
| meta | 0.957 | 1.000 | 0.978 | 22 |
| research | 0.917 | 1.000 | 0.957 | 22 |
| smalltalk | 1.000 | 1.000 | **1.000** | 10 |

### Step 14 — Per-example failure audit

**File:** `results/eval_2026-07-09.md`

All 8 failures were read individually. **Only 2 of 8 were genuine model errors.**

- **2 = truncation, not malformed JSON.** Both "parse errors" were *valid* JSON cut off at `max_new_tokens=256` — the model wrote long numbered-step `reasoning` fields that ran past the budget. One ends mid-word at `"capi"`. Fix: raise to 384–512. A config change, not a reliability problem.
- **4 = label noise, model arguably correct.** Queries labeled `clarify` that the model called `research` or `lookup` — e.g. *"What was the R&D spend of the highest-revenue tech company in 2024?"* is unambiguous multi-hop reasoning; `research` is the better answer and there is nothing to clarify. The labeler **over-applied `clarify`**, and that noise is presumably in the training split too.
- **2 = the real failure.** Corpus-boundary overstep: `decline → lookup`. The model treated out-of-coverage queries as answerable. Directly traceable to Step 7 — only 3 corpus-boundary examples were injected into ~1,000, so the student barely learned that boundary.

The interesting positive result: **`compute` vs `lookup` — the boundary expected to be hardest — was 21/21 with zero leakage.** The reason is upstream. Those disagreements were resolved during labeling (Step 6) plus the surgical overrides (Step 7), so by training time the boundary was *consistent*. A consistent boundary is learned sharply. Label consistency mattered more than label volume.

---

## The compressed recipe

| # | Step | Script | Out |
|---|---|---|---|
| 1 | Taxonomy as code | `prompts/directive_labeler.py` | 7 codes, 4 sub-reasons |
| 2 | Teacher labeling prompt | same | validated 100% on hand eval |
| 3 | 3-layer generator prompt | `prompts/synthetic_query_generator.py` | base + per-code guidance |
| 4 | Generate | `generate_synthetic_queries.py` | 1,400 raw (~$1.50) |
| 5 | Dedup | `dedup_synthetic_queries.py` | 998 unique (−29%) |
| 6 | Label independently | `label_synthetic_queries.py` | 998 labeled + disagreements (~$11) |
| 7 | Review + override | `apply_label_overrides.py` | 3 corpus-boundary fixes |
| 8 | Stratified split | `build_splits.py` | 696 / 146 / 156, 4 assertions |
| 9 | Publish dataset | `push_splits_to_hf.py` | `AlHindi/findirector-splits` |
| 10 | Freeze hyperparams | `training_config.py` | r=16, α=32, all-linear, lr 3e-4 |
| 11 | Train script | `train_directive_model.py` | SFTTrainer, bf16, eb=32, 3 epochs |
| 12 | Run on L4 | `train_on_colab.ipynb` (generated) | adapter → HF Hub, 30–90 min |
| 13 | Eval by generation | `evaluate_directive_model.py` | 94.87%, per-example detail saved |
| 14 | Failure audit | `results/eval_2026-07-09.md` | 2 real errors of 8 |

**Total cost of the data:** ~$12.50 of API calls plus a few dollars of Colab compute.

---

## Appendix — What this maps to in the config-driven platform

Every hard-coded value above is a config field, and every script is a pipeline stage. The natural decomposition for a SageMaker-based low-code platform:

**Recipe: `supervised-lora-classifier`**

| Platform concept | Comes from |
|---|---|
| **Taxonomy / schema definition** | Step 1 — user-supplied label set + output JSON schema |
| **Teacher prompt template** | Steps 2–3 — the three-layer structure (base scaffold + per-class guidance) is a fill-in-the-blanks form, not free text |
| **Synthetic data job** | Step 4 — SageMaker Processing job; params: `target_per_class`, `batch_size`, `teacher_model` |
| **Dedup + label + review** | Steps 5–7 — Processing jobs; the disagreement file becomes a **human-in-the-loop review UI** (this is the single highest-value screen in the product) |
| **Split job** | Step 8 — params: ratios, seed, stratify key; the 4 sanity checks become **hard gates** |
| **Dataset registry** | Step 9 — SageMaker Feature Store / S3 + versioned manifest instead of the HF Hub |
| **Training config** | Steps 10–11 — the `training_config.py` module *is* the config schema; expose r / α / dropout / target modules / lr / batch / epochs with the recipe's defaults pre-filled |
| **Training job** | Step 12 — SageMaker Training job (ml.g6.xlarge ≈ L4); replaces the Colab notebook entirely |
| **Model registry** | Step 12 — SageMaker Model Registry for the adapter artifact |
| **Eval job + report** | Steps 13–14 — Processing job emitting metrics JSON + per-example JSONL; the per-example file drives a **failure-explorer UI** |
| **Experiment tracking** | W&B → SageMaker Experiments (or keep W&B) |

Three lessons from this run that should be **structural** in the platform, not documentation:

1. **Generation and labeling must be separate calls with separate prompts.** Store both `intended_class` and `predicted_class`; the disagreement rate is your free taxonomy-quality metric.
2. **Prompt shape must be captured once and reused by train *and* eval.** The single biggest silent-failure mode here is train/inference prompt drift. Serialize the prompt template into the model artifact.
3. **Domain constraints the teacher can't know need an explicit injection step.** The 2 real errors trace directly to only 3 corpus-boundary examples existing. The platform should have a first-class "constraint examples" input, not a manual override script.
