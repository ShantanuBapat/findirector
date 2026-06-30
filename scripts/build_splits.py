"""
Build train/val/test splits from labeled synthetic queries.

Strategy:
- 70/15/15 stratified split (proportions preserved per code)
- Random seed 42 for reproducibility
- Label source: predicted_code (Phase 5 labeler verdicts + 3 overrides)
- Output: instruction-following format for LoRA fine-tuning

Output files:
- data/synthetic/splits/train.jsonl
- data/synthetic/splits/val.jsonl
- data/synthetic/splits/test.jsonl

Each record:
{
  "instruction": "Classify the following financial query...",
  "input": "<query text>",
  "output": "<JSON of action_code + params + reasoning>",
  "_meta": {
    "intended_code": "<generator intent>",
    "predicted_code": "<labeler verdict>",
    "agrees": <bool>
  }
}

Sanity checks:
- Total preserved (998 in = 998 out)
- No leakage (no query appears in multiple splits)
- Per-code proportions match the target ratio
- Each code has at least 1 example in each split

Usage:
    python -m scripts.build_splits
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from prompts.directive_labeler import ACTION_CODES


# Configuration
LABELED_DIR = Path("data/synthetic/labeled")
SPLITS_DIR = Path("data/synthetic/splits")
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

INSTRUCTION = (
    "Classify the following financial query into one of 7 action codes "
    "(smalltalk, meta, lookup, compute, research, clarify, decline). "
    "Output JSON with action_code, params, and reasoning."
)


def load_all_labeled() -> list[dict]:
    """Load all labeled records from the 7 per-code files."""
    all_records = []
    for code in ACTION_CODES:
        path = LABELED_DIR / f"{code}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} missing — skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    all_records.append(record)
    return all_records


def stratified_split(
    records: list[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_by: str = "predicted_code",
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split: same proportions per `stratify_by` value across splits.

    Returns (train, val, test) as lists of records.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        "Fractions must sum to 1"

    # Group records by stratification key
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = record.get(stratify_by, "UNKNOWN")
        groups[key].append(record)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    rng = random.Random(seed)

    # For each group, shuffle and split proportionally
    for key in sorted(groups.keys()):
        group_records = list(groups[key])  # copy
        rng.shuffle(group_records)

        n = len(group_records)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # Test takes the remainder so we don't lose records to rounding
        n_test = n - n_train - n_val

        train.extend(group_records[:n_train])
        val.extend(group_records[n_train:n_train + n_val])
        test.extend(group_records[n_train + n_val:])

    return train, val, test


def to_instruction_format(record: dict) -> dict:
    """
    Convert a labeled record into the instruction-following training format.
    """
    output_obj = {
        "action_code": record["predicted_code"],
        "params": record.get("params", {}),
        "reasoning": record.get("reasoning", ""),
    }

    return {
        "instruction": INSTRUCTION,
        "input": record["query"],
        "output": json.dumps(output_obj),
        "_meta": {
            "intended_code": record.get("intended_code"),
            "predicted_code": record.get("predicted_code"),
            "agrees": record.get("agrees"),
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def check_no_leakage(splits: dict[str, list[dict]]) -> tuple[bool, str]:
    """Verify no query appears in more than one split."""
    seen_in: dict[str, str] = {}  # query -> split_name
    for split_name, records in splits.items():
        for r in records:
            query = r["input"]
            if query in seen_in:
                return False, (
                    f"LEAKAGE: query {query!r} in both "
                    f"{seen_in[query]} and {split_name}"
                )
            seen_in[query] = split_name
    return True, ""


def check_total_preserved(
    input_count: int, splits: dict[str, list[dict]]
) -> tuple[bool, str]:
    """Verify total record count is preserved."""
    out_count = sum(len(r) for r in splits.values())
    if out_count != input_count:
        return False, (
            f"COUNT MISMATCH: input={input_count} output={out_count}"
        )
    return True, ""


def check_stratification(
    splits: dict[str, list[dict]],
    target_fracs: dict[str, float],
    tolerance: float = 0.05,
) -> tuple[bool, list[str]]:
    """Verify per-code proportions are within tolerance of target."""
    issues = []

    # Compute overall per-code totals (across all splits) for ratio check
    overall_counts: Counter = Counter()
    for split_records in splits.values():
        for r in split_records:
            overall_counts[r["_meta"]["predicted_code"]] += 1

    for split_name, records in splits.items():
        target = target_fracs[split_name]
        per_code_in_split: Counter = Counter()
        for r in records:
            per_code_in_split[r["_meta"]["predicted_code"]] += 1

        for code in ACTION_CODES:
            total_for_code = overall_counts.get(code, 0)
            if total_for_code == 0:
                continue
            in_split = per_code_in_split.get(code, 0)
            observed_frac = in_split / total_for_code
            if abs(observed_frac - target) > tolerance:
                issues.append(
                    f"  STRATIFICATION ISSUE: code={code} split={split_name} "
                    f"target={target:.2f} observed={observed_frac:.2f}"
                )

    return len(issues) == 0, issues


def check_min_per_class(
    splits: dict[str, list[dict]], min_count: int = 1
) -> tuple[bool, list[str]]:
    """Verify each split has at least `min_count` examples per code."""
    issues = []
    for split_name, records in splits.items():
        per_code = Counter(r["_meta"]["predicted_code"] for r in records)
        for code in ACTION_CODES:
            if per_code.get(code, 0) < min_count:
                issues.append(
                    f"  MIN-PER-CLASS ISSUE: code={code} split={split_name} "
                    f"has {per_code.get(code, 0)} examples (min={min_count})"
                )
    return len(issues) == 0, issues


def main() -> int:
    if not LABELED_DIR.exists():
        print(f"ERROR: {LABELED_DIR} not found. Run Phase 5 labeling first.")
        return 1

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading labeled records from {LABELED_DIR}/")
    records = load_all_labeled()
    print(f"Loaded {len(records)} labeled records.")

    # Split
    print(f"\nSplitting {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%} "
          f"stratified by predicted_code (seed={SEED})")
    train_raw, val_raw, test_raw = stratified_split(
        records, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, SEED
    )

    # Convert to instruction-following format
    train = [to_instruction_format(r) for r in train_raw]
    val = [to_instruction_format(r) for r in val_raw]
    test = [to_instruction_format(r) for r in test_raw]

    splits = {"train": train, "val": val, "test": test}
    target_fracs = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}

    # Sanity checks
    print(f"\nRunning sanity checks...")
    all_passed = True

    ok, msg = check_total_preserved(len(records), splits)
    print(f"  Total preserved:     {'✓' if ok else '✗ ' + msg}")
    all_passed = all_passed and ok

    ok, msg = check_no_leakage(splits)
    print(f"  No leakage:          {'✓' if ok else '✗ ' + msg}")
    all_passed = all_passed and ok

    ok, issues = check_stratification(splits, target_fracs)
    print(f"  Stratification:      {'✓' if ok else '✗ (see below)'}")
    if not ok:
        for issue in issues:
            print(issue)
    all_passed = all_passed and ok

    ok, issues = check_min_per_class(splits, min_count=2)
    print(f"  Min 2 per class:     {'✓' if ok else '✗ (see below)'}")
    if not ok:
        for issue in issues:
            print(issue)
    all_passed = all_passed and ok

    if not all_passed:
        print("\nERROR: Some sanity checks failed. Refusing to write splits.")
        return 1

    # Write outputs
    print(f"\nWriting splits to {SPLITS_DIR}/")
    for split_name in ["train", "val", "test"]:
        out_path = SPLITS_DIR / f"{split_name}.jsonl"
        write_jsonl(out_path, splits[split_name])
        print(f"  {out_path.name}: {len(splits[split_name])} records")

    # Summary table — counts per code per split
    print(f"\nPer-code counts:")
    print(f"  {'code':12s}  {'train':>6s}  {'val':>5s}  {'test':>5s}  "
          f"{'total':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")

    for code in ACTION_CODES:
        n_train = sum(1 for r in train if r["_meta"]["predicted_code"] == code)
        n_val = sum(1 for r in val if r["_meta"]["predicted_code"] == code)
        n_test = sum(1 for r in test if r["_meta"]["predicted_code"] == code)
        total = n_train + n_val + n_test
        print(f"  {code:12s}  {n_train:>6d}  {n_val:>5d}  {n_test:>5d}  "
              f"{total:>6d}")

    n_train_total = len(train)
    n_val_total = len(val)
    n_test_total = len(test)
    grand_total = n_train_total + n_val_total + n_test_total

    print(f"  {'-'*12}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")
    print(f"  {'TOTAL':12s}  {n_train_total:>6d}  {n_val_total:>5d}  "
          f"{n_test_total:>5d}  {grand_total:>6d}")

    print(f"\nDone. Splits ready for fine-tuning (Session 2.5).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
