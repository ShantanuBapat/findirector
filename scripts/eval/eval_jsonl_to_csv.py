# scripts/eval/eval_jsonl_to_csv.py
#
# WHAT THIS DOES:
#   Converts the draft eval set (JSONL) to a flat CSV for human review in a
#   spreadsheet. Each eval case becomes one row; nested source fields are
#   flattened into columns. Adds a "keep" column (default "yes") so you can mark
#   rows to cull by setting it to "no". You review/edit in a spreadsheet, save,
#   then a companion script converts the cleaned CSV back to JSONL.
#
# INPUT:  --in  (draft JSONL, default data/eval/eval_set_draft.jsonl)
# OUTPUT: --out (CSV, default data/eval/eval_set_review.csv)
#
# COLUMNS (review these):
#   keep              - "yes"/"no"; set "no" to drop the row
#   id                - case id
#   question          - editable
#   expected_answer   - editable (the main thing to sanity-check)
#   answer_type       - "numeric"/"text" (editable)
#   question_type     - category tag
#   src_ticker / src_fiscal_year / src_section / src_chunk_id
#                     - the source label (read-only reference; don't edit)

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/eval/eval_set_draft.jsonl")
    ap.add_argument("--out", default="data/eval/eval_set_review.csv")
    args = ap.parse_args()

    rows = []
    for line in Path(args.inp).open():
        if not line.strip():
            continue
        c = json.loads(line)
        src = c.get("source", {})
        rows.append({
            "keep": "yes",
            "id": c["id"],
            "question": c["question"],
            "expected_answer": c["expected_answer"],
            "answer_type": c["answer_type"],
            "question_type": c.get("question_type", ""),
            "src_ticker": src.get("ticker", ""),
            "src_fiscal_year": src.get("fiscal_year", ""),
            "src_section": src.get("section", ""),
            "src_chunk_id": src.get("chunk_id", ""),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["keep", "id", "question", "expected_answer", "answer_type",
              "question_type", "src_ticker", "src_fiscal_year", "src_section",
              "src_chunk_id"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out_path}")
    print("Open in a spreadsheet, review, set keep=no to cull, save as CSV.")


if __name__ == "__main__":
    main()