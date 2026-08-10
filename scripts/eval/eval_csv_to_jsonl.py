# scripts/eval/eval_csv_to_jsonl.py
#
# WHAT THIS DOES:
#   Converts the reviewed eval CSV back into the final EvalCase JSONL. Keeps only
#   rows where keep == "yes" (culled rows are dropped), reconstructs the nested
#   source object from the flat src_* columns, and writes the trusted eval set
#   that the harness will run against.
#
# INPUT:  --in  (reviewed CSV, default data/eval/eval_set_review.csv)
# OUTPUT: --out (final JSONL, default data/eval/eval_set.jsonl)
#
# NOTE: run this AFTER the human-review pass. Rows with keep != "yes" are
#   excluded. The src_* columns are read back into source (unchanged); the
#   editable columns (question / expected_answer / answer_type) reflect any
#   corrections made during review.

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from scripts.eval.eval_types import EvalCase, EvalSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/eval/eval_set_review.csv")
    ap.add_argument("--out", default="data/eval/eval_set.jsonl")
    args = ap.parse_args()

    kept, dropped = 0, 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.inp).open() as f, out_path.open("w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("keep", "").strip().lower() != "yes":
                dropped += 1
                continue
            case = EvalCase(
                id=row["id"],
                question=row["question"],
                expected_answer=row["expected_answer"],
                answer_type=row["answer_type"],
                source=EvalSource(
                    ticker=row["src_ticker"],
                    fiscal_year=int(row["src_fiscal_year"]),
                    section=row["src_section"] or None,
                    chunk_id=row["src_chunk_id"] or None,
                ),
                directive_params={
                    "company": row["src_ticker"],
                    "year": int(row["src_fiscal_year"]),
                },
                question_type=row.get("question_type") or "lookup_fact",
            )
            out.write(json.dumps(asdict(case)) + "\n")
            kept += 1

    print(f"kept {kept}, dropped {dropped} -> {out_path}")


if __name__ == "__main__":
    main()