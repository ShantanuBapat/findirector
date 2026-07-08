"""
Configuration for LoRA fine-tuning of the FinDirector directive model.

All hyperparameters and paths for training are defined here.
The training script imports from this module.

Design decisions locked in Session 2.4 Phase 2.
"""

from pathlib import Path


# ============================================================================
# Model configuration (Session 2.4 Phase 2, Decision 1)
# ============================================================================

# Base model — Qwen 2.5 7B Instruct
# Load from Hugging Face Hub; auto-downloads to ~/.cache/huggingface
BASE_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"

# Precision for loading the base model (Decision 1)
# bfloat16 is the modern default: same numerical range as fp32, half the memory
BASE_MODEL_DTYPE: str = "bfloat16"


# ============================================================================
# LoRA configuration (Decision 2 + 3)
# ============================================================================

# LoRA rank — controls adaptation capacity (Decision 2)
# 16 is sweet spot for classification tasks
LORA_R: int = 16

# LoRA alpha — scales the update magnitude
# 2:1 alpha:rank ratio is the modern default
LORA_ALPHA: int = 32

# LoRA dropout — regularization on the adapter
LORA_DROPOUT: float = 0.05

# Target modules — which layers get LoRA adapters (Decision 3)
# "All linear layers" — modern best practice
LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================================
# Training hyperparameters (Decision 4)
# ============================================================================

# Optimizer learning rate — LoRA can handle higher LR than full FT
LEARNING_RATE: float = 3e-4

# Weight decay — L2 regularization on trainable parameters
WEIGHT_DECAY: float = 0.01

# Batch size and gradient accumulation
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
PER_DEVICE_TRAIN_BATCH_SIZE: int = 4
GRADIENT_ACCUMULATION_STEPS: int = 8
EFFECTIVE_BATCH_SIZE: int = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Same batch size for evaluation
PER_DEVICE_EVAL_BATCH_SIZE: int = 4

# Number of epochs — 3 is starting point; can extend if val loss still improving
NUM_TRAIN_EPOCHS: int = 3

# Warmup ratio — first 10% of steps ramp up LR linearly
WARMUP_RATIO: float = 0.1

# LR schedule — cosine decay after warmup
LR_SCHEDULER_TYPE: str = "cosine"

# Optimizer choice
OPTIMIZER: str = "adamw_torch"


# ============================================================================
# Tokenization configuration (Decision 5)
# ============================================================================

# Maximum sequence length
# Our data tops out around 300 tokens; 512 gives safe headroom
MAX_SEQ_LENGTH: int = 512

# Padding side for the tokenizer
# Right-padding is standard for training; masked properly by attention mask
TOKENIZER_PADDING_SIDE: str = "right"


# ============================================================================
# Logging and checkpointing configuration (Decision 6)
# ============================================================================

# Experiment tracking backend
REPORT_TO: str = "wandb"

# Log training loss every N optimizer steps
LOGGING_STEPS: int = 5

# Run evaluation on val set at every epoch
EVALUATION_STRATEGY: str = "epoch"

# Save a checkpoint every epoch
SAVE_STRATEGY: str = "epoch"

# Keep only the 2 most recent checkpoints
SAVE_TOTAL_LIMIT: int = 2

# After training, load the checkpoint with the best val loss
LOAD_BEST_MODEL_AT_END: bool = True
METRIC_FOR_BEST_MODEL: str = "eval_loss"
GREATER_IS_BETTER: bool = False


# ============================================================================
# Reproducibility
# ============================================================================

# Random seed for training (independent of data-split seed)
TRAINING_SEED: int = 42


# ============================================================================
# Paths (relative to project root)
# ============================================================================

# Where our splits live
DATA_DIR: Path = Path("data/synthetic/splits")
TRAIN_FILE: Path = DATA_DIR / "train.jsonl"
VAL_FILE: Path = DATA_DIR / "val.jsonl"
TEST_FILE: Path = DATA_DIR / "test.jsonl"

# Where training outputs go
OUTPUT_DIR: Path = Path("outputs/qwen-findirector-lora")


# ============================================================================
# W&B project configuration
# ============================================================================

# W&B project name (visible in wandb.ai dashboard)
WANDB_PROJECT: str = "findirector-directive-model"

# Run name will be auto-generated with a timestamp if not overridden
WANDB_RUN_NAME: str | None = None
