# scripts/eval/eval_types.py
#
# WHAT THIS DEFINES:
#   The EvalCase schema — the shape of a single evaluation item. Every case in
#   the corpus-grounded eval set is one of these. The fields are chosen so ONE
#   set can drive BOTH retrieval eval (via the labeled source chunk) and
#   generation eval (via expected_answer + answer_type), and so scores can be
#   sliced by category.
#
# FIELDS:
#   id                - unique id for the case (to reference specific results).
#   question          - the user-facing question fed to the pipeline.
#   expected_answer   - the known-correct answer (ground truth).
#   answer_type       - "numeric" | "text"; decides the scoring method
#                       (near-match for numbers, LLM-as-judge for prose).
#   source            - EvalSource: which chunk the answer comes from
#                       (ticker/fiscal_year/section/chunk_id) — the RETRIEVAL
#                       label used for hit-rate / MRR.
#   directive_params  - the {company, year, fact_requested} a correct router
#                       should emit; lets us test routing and/or feed retrieval
#                       directly to isolate a stage.
#   question_type     - category tag ("lookup_fact", "definition", ...) for
#                       slicing scores; extensible as compute/research land.
#
# WHY A dataclass (not a bare dict):
#   Named, typed fields catch mistakes early and make the generator, the
#   human-verification step, and the harness all agree on the shape.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalSource:
    """Where the ground-truth answer lives — the retrieval-eval label."""
    ticker: str
    fiscal_year: int
    section: Optional[str] = None
    chunk_id: Optional[str] = None     # the specific chunk, if known


@dataclass
class EvalCase:
    """One evaluation item: a question with its known answer and source."""
    id: str
    question: str
    expected_answer: str
    answer_type: str                    # "numeric" | "text"
    source: EvalSource
    directive_params: dict = field(default_factory=dict)
    question_type: str = "lookup_fact"  # extensible category tag