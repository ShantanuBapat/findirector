"""
LoRA fine-tuning script for the FinDirector directive model.

Trains Qwen 2.5 7B Instruct with LoRA adapters on our synthetic
labeled query dataset. All hyperparameters are defined in
scripts/training_config.py.

Pipeline:
    1. Load tokenizer and base model in bfloat16
    2. Configure LoRA adapters (r=16, alpha=32, all linear layers)
    3. Load train/val JSONL splits into HF Dataset format
    4. Set up SFTTrainer with proper loss masking
    5. Run training loop (3 epochs, effective batch 32)
    6. Save the trained LoRA adapter

Usage:
    python -m scripts.train_directive_model

For real training, run on Colab Pro L4 GPU (24 GB) or better.
Mac M4 MPS backend works for smoke tests but will be slow for a full run.
"""

import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from scripts.training_config import (
    BASE_MODEL_NAME,
    BASE_MODEL_DTYPE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_TRAIN_EPOCHS,
    WARMUP_RATIO,
    LR_SCHEDULER_TYPE,
    OPTIMIZER,
    MAX_SEQ_LENGTH,
    TOKENIZER_PADDING_SIDE,
    REPORT_TO,
    LOGGING_STEPS,
    EVALUATION_STRATEGY,
    SAVE_STRATEGY,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    METRIC_FOR_BEST_MODEL,
    GREATER_IS_BETTER,
    TRAINING_SEED,
    TRAIN_FILE,
    VAL_FILE,
    OUTPUT_DIR,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)


# ============================================================================
# Helper: get the training device
# ============================================================================

def get_device() -> str:
    """
    Return the best available device for training.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ============================================================================
# Data loading
# ============================================================================

def load_jsonl_as_dataset(jsonl_path: Path) -> Dataset:
    """
    Load a JSONL file into a HuggingFace Dataset.

    Each record should have 'instruction', 'input', 'output' fields.
    Additional fields (like '_meta') are preserved but ignored by training.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Data file not found: {jsonl_path}")

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return Dataset.from_list(records)


def format_for_sft(record: dict) -> dict:
    """
    Convert an instruction-following record into the format SFTTrainer expects.

    SFTTrainer with formatting_func expects a single 'text' field with the
    fully-formatted training example.

    We build the conversation manually rather than using apply_chat_template
    at this stage, so we have precise control over what tokens contribute to loss.
    """
    # Concatenate instruction and input for the user message
    user_content = f"{record['instruction']}\n\nQuery: {record['input']}"
    assistant_content = record["output"]

    # Build the conversation in messages format
    # SFTTrainer will apply the chat template during tokenization
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ============================================================================
# Model and tokenizer setup
# ============================================================================

def load_tokenizer():
    """Load and configure the Qwen 2.5 7B tokenizer."""
    print(f"Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )

    # Qwen doesn't always set pad_token; use eos_token as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Right-padding is standard for training
    tokenizer.padding_side = TOKENIZER_PADDING_SIDE

    return tokenizer


def load_base_model():
    """Load Qwen 2.5 7B Instruct in bfloat16."""
    print(f"Loading base model: {BASE_MODEL_NAME}")
    print(f"  Precision: {BASE_MODEL_DTYPE}")

    device = get_device()
    print(f"  Device: {device}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[BASE_MODEL_DTYPE]

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    # Disable KV caching during training (required by HF)
    model.config.use_cache = False

    return model


def configure_lora(model):
    """Wrap the base model with LoRA adapters."""
    print(f"Configuring LoRA:")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    return model


# ============================================================================
# Trainer setup
# ============================================================================

def build_training_config() -> SFTConfig:
    """Build the SFTConfig with all our training hyperparameters."""
    output_path = OUTPUT_DIR.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    return SFTConfig(
        # Output
        output_dir=str(output_path),

        # Training loop
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIMIZER,

        # Precision — bf16 for training with mixed precision
        bf16=True,

        # Logging
        logging_steps=LOGGING_STEPS,
        report_to=REPORT_TO,

        # Evaluation
        eval_strategy=EVALUATION_STRATEGY,

        # Checkpointing
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,

        # Reproducibility
        seed=TRAINING_SEED,

        # Sequence length
        max_length=MAX_SEQ_LENGTH,

        # Disable removing unused columns (we have _meta)
        remove_unused_columns=False,

        # W&B
        run_name=WANDB_RUN_NAME,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    print("=" * 80)
    print("FinDirector — LoRA Fine-Tuning")
    print("=" * 80)
    print()

    # Set up environment for W&B
    import os
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    # Load data
    print("Loading data...")
    t0 = time.time()
    train_dataset = load_jsonl_as_dataset(TRAIN_FILE)
    val_dataset = load_jsonl_as_dataset(VAL_FILE)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    # Format for SFT
    print("Formatting data for SFT...")
    t0 = time.time()
    train_dataset = train_dataset.map(format_for_sft)
    val_dataset = val_dataset.map(format_for_sft)
    print(f"  Formatted in {time.time() - t0:.1f}s")
    print()

    # Load tokenizer
    t0 = time.time()
    tokenizer = load_tokenizer()
    print(f"  Loaded tokenizer in {time.time() - t0:.1f}s")
    print()

    # Load base model
    t0 = time.time()
    model = load_base_model()
    print(f"  Loaded base model in {time.time() - t0:.1f}s")
    print()

    # Wrap with LoRA
    t0 = time.time()
    model = configure_lora(model)
    print(f"  Configured LoRA in {time.time() - t0:.1f}s")
    print()

    # Build training config
    training_args = build_training_config()

    # Build trainer
    print("Building trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    print()

    # Train
    print("Starting training...")
    print("=" * 80)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print()
    print("=" * 80)
    print(f"Training complete in {train_time / 60:.1f} minutes")
    print()

    # Save final model (best checkpoint via load_best_model_at_end)
    print(f"Saving final model to {OUTPUT_DIR}/")
    trainer.save_model()
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
