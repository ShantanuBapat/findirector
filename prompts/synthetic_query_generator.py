"""
Synthetic query generator for FinDirector directive model training.

Generates realistic, diverse training queries for each of the 7 action codes.
The output queries are then labeled by `prompts/directive_labeler.py` to
produce the labeled training set used to fine-tune the directive model.

Architecture (Option C — hybrid):
  Layer 1: BASE_INSTRUCTIONS  — shared scaffold (role, corpus, diversity, format)
  Layer 2: PER_CODE_GUIDANCE  — per-code specialization (dict keyed by code)
  Layer 3: build_generation_prompt(code) — composes Layer 1 + Layer 2

Public API:
  - build_generation_prompt(code: str) -> str
  - generate_queries(client, code: str, n: int) -> list[str]
"""

import json
from typing import Final

from anthropic import Anthropic

from prompts.directive_labeler import ACTION_CODES


# =============================================================================
# Layer 1 — Shared scaffold
# =============================================================================

BASE_INSTRUCTIONS: Final[str] = """\
ROLE
====
You are a synthetic data generator for FinDirector, a financial Q&A system
that answers questions about SEC 10-K filings. Your job is to produce
realistic, diverse training queries that match a specified action code.

You generate queries; you do NOT classify, answer, or judge them. The
classification happens in a separate downstream pipeline.


CORPUS CONSTRAINTS
==================
The queries you generate will be used to train a model that answers
questions about a specific 10-K corpus. Stay within this corpus:

Available companies (use ticker or company name, vary both):
- Big Tech: Apple (AAPL), Microsoft (MSFT), Alphabet/Google (GOOGL),
  Amazon (AMZN), Meta/Facebook (META), NVIDIA (NVDA)
- Finance: JPMorgan (JPM), Bank of America (BAC), Berkshire Hathaway
  (BRK-B), Visa (V)
- Healthcare: Johnson & Johnson (JNJ), UnitedHealth (UNH), Pfizer (PFE)
- Consumer: Walmart (WMT), Coca-Cola (KO), Procter & Gamble (PG)
- Energy: ExxonMobil (XOM), Chevron (CVX)
- Industrial/Auto: Tesla (TSLA), Boeing (BA)

Available years: 2022, 2023, 2024 (most queries should use these)
Time expressions are fine: "last year", "fiscal 2023", "FY24", "recently"

Do NOT generate queries about companies not in this list (e.g., Netflix,
Spotify, Disney). Do NOT reference years before 2022 or after 2024.


DIVERSITY REQUIREMENTS
======================
Within each generation batch of N queries, ensure variation across:

1. Companies — distribute across sectors, don't pile up on AAPL/MSFT
2. Time references — mix specific years, relative time ("last year"),
   fiscal year notation ("FY23", "fiscal 2024")
3. Phrasing formality — mix of:
   - Casual: "what's apple doing for r&d these days"
   - Neutral: "What was Apple's R&D spending in 2023?"
   - Formal: "Could you provide Apple's research and development
     expenditure for fiscal year 2023?"
4. Question structure — mix interrogatives (what/how/when), imperatives
   ("tell me about X", "compare X and Y"), and partial questions
5. Length — some short (5-10 words), some medium (15-25 words),
   occasional longer (30+ words)


ANTI-PATTERNS (avoid these in every batch)
===========================================
- NO multi-part queries ("What was X's revenue AND R&D?")
- NO repeated queries within a batch (vary the surface form even when
  the underlying concept is similar)
- NO meta-commentary in the queries themselves ("this is a financial
  question:", "I want to know:")
- NO queries that reference things outside the 10-K corpus (stock price
  movements, news events, analyst opinions)
- NO queries that ARE labels rather than queries ("this is a lookup query:")


OUTPUT FORMAT
=============
Return a single JSON array of N strings. No prose before or after. No
markdown code fences. No commentary. Just the JSON array.

Format:
[
  "First query string",
  "Second query string",
  ...
  "Nth query string"
]

Each string is a complete, standalone user query — exactly as a user
would type it. No labels, no metadata, no JSON objects within the array.
"""


# =============================================================================
# Layer 2 — Per-code specialization
# =============================================================================

PER_CODE_GUIDANCE: Final[dict[str, str]] = {
    "smalltalk": """\
SMALLTALK GENERATION GUIDANCE
==============================

Defining traits:
Smalltalk queries are purely social/conversational with NO question,
NO request for information, NO retrieval needed. They are what users
say when they're being human, not when they're querying the system.

Sub-types to include in each batch:
1. Opening greetings: "Hi", "Hey there", "Good morning"
2. Closing/gratitude: "Thanks!", "That helps", "Got it, appreciate it"
3. Conversational reactions: "Oh interesting", "That makes sense",
   "Hmm okay"
4. Mid-conversation acknowledgments: "Yeah", "Right", "Okay"
5. Polite framing without a question: "I have a question coming up",
   "Hold on, let me think"
6. Mild frustration or confusion as social signal (no actionable request):
   "Wait what", "I'm confused"
7. Casual remarks with financial-adjacent words BUT no question:
   "That Apple report was interesting", "Earnings season is wild"

Edge cases to specifically include:
- Smalltalk that MENTIONS a company but asks nothing
  ("Apple's been busy lately") — tests that the directive model
  doesn't auto-route on entity presence
- Smalltalk that follows a financial answer but is just social
  ("ok thanks that's what I needed")
- Short responses (1-3 words) AND medium social statements (10-15 words)

Specifically AVOID:
- Any query with a question mark followed by something the system
  could actually answer
- Any imperative like "tell me about X" (that's not smalltalk)
- Any query starting with "what", "when", "how", "why" UNLESS it's
  rhetorical/social ("how about that")
- Greetings that bundle a real question ("Hi, what was Apple's revenue?")
  — split these into smalltalk-then-followup, but here just emit the
  pure greeting
""",

    "meta": """\
META GENERATION GUIDANCE
========================

Defining traits:
Meta queries ask ABOUT FinDirector (its capabilities, scope, behavior,
or about the financial-reporting domain in general) rather than asking
the system FOR specific financial information from filings.

Sub-types to include in each batch:
1. Capability questions: "Can you read quarterly filings?",
   "Do you handle 8-K reports?", "What's your data cutoff?"
2. Scope questions: "Which companies do you cover?",
   "Do you have data on European companies?",
   "Can you analyze private companies?"
3. Identity/behavior questions: "Are you a financial advisor?",
   "What kind of system are you?", "How do you handle uncertainty?"
4. Domain concept questions (about SEC filings generally, NOT a
   specific filing): "What's the difference between 10-K and 10-Q?",
   "What does MD&A stand for?", "What is a risk factor section?"
5. Usage/how-to questions: "How should I phrase a question?",
   "Can I ask follow-up questions?", "What format do you respond in?"
6. Limitations questions: "What can't you answer?",
   "Are your answers audited?", "How recent is your data?"

Edge cases to specifically include:
- Capability questions that sound financial but ARE meta:
  "Do you know about ESG metrics?" (capability question → meta)
- Domain questions that name a specific company but stay conceptual:
  "When does Apple typically file their 10-K?" (asking about filing
  timing as a general concept, NOT Apple's specific revenue)
- "How" questions about the system's process:
  "How do you decide which sections to look at?"

Specifically AVOID:
- Questions that look meta but actually request specific filing data:
  "What does Apple say about cloud?" (this is `lookup`, not `meta`)
- Existential/philosophical questions: "Are you sentient?",
  "Do you have feelings?" (technically meta but useless for our domain)
- Questions about Anthropic/Claude/the underlying model:
  "What model are you?", "Who made you?" (out-of-scope for FinDirector
  meta — we want questions about FinDirector itself)
- Bundled meta + lookup: "What can you do? Also, what's Apple's revenue?"
  (split would be a different code; here generate pure meta)
""",

    "lookup": """\
LOOKUP GENERATION GUIDANCE
==========================

Defining traits:
Lookup queries request a SINGLE FACT or DEFINITION from ONE company's
filings. Single document, single retrieval. No arithmetic, no multi-doc
synthesis, no advice, no prediction.

Sub-types to include in each batch:
1. Specific number requests: "What was Apple's R&D in 2023?"
   Variations: total revenue, net income, R&D, operating expenses,
   capex, headcount, segment revenue, gross profit
2. Definitional/conceptual queries: "What does Microsoft mean by
   'commercial cloud'?", "How does Tesla define its automotive segment?"
3. Disclosure queries: "What risks does JPMorgan list in their 2023 10-K?",
   "What's in Walmart's strategy section?"
4. Section-specific queries: "What's in the MD&A of Apple's 10-K?",
   "Tell me about Pfizer's pipeline mentioned in their 2023 filing"
5. Mention/inclusion queries: "Does Boeing discuss supply chain in their
   2024 10-K?", "Does Coca-Cola mention sustainability in fiscal 2023?"
6. Quote-extraction style: "What did Tim Cook say about AI in Apple's
   2024 annual report?", "What's UnitedHealth's stated mission?"

Variation dimensions to hit:
- Company: distribute across all 20 tickers, NOT just AAPL/MSFT
- Year: mix 2022, 2023, 2024 — slightly favor 2023 (most common
  reference year in practice)
- Phrasing: mix question-form ("What was..."), imperative ("Tell me..."),
  declarative-with-question-mark ("Apple's revenue 2023?")
- Metric type: numbers, concepts, sections, mentions, quotes
- Tense: "What was..." (past), "What does Apple say about..." (present),
  "In Apple's 2023 filing..." (locative)

Edge cases to specifically include:
- Conceptual queries that LOOK complex but are single-doc lookups:
  "What does Apple mean by 'Services'?" (looks deep, is lookup)
  "How does Microsoft describe its risk factors related to AI?"
- Queries with implicit time reference: "What was Apple's R&D last year?"
  (resolvable if context is clear)
- Tickerless company names: "What was Berkshire Hathaway's revenue?"
  (BRK-B in our corpus but users may not know the ticker)
- Casual/informal phrasing of straightforward requests:
  "apple revenue 2023?", "how much did Tesla make last year"

Specifically AVOID:
- Multi-company comparisons: "Apple vs Microsoft revenue" (→ research)
- Multi-year trends: "Apple R&D 2020-2024" (→ research)
- Arithmetic implied: "Apple R&D as percentage of revenue" (→ compute)
- Investment angle: "Is Apple's R&D spending too high?" (→ decline,
  investment_advice)
- Prediction angle: "Will Apple increase R&D next year?" (→ decline,
  prediction)
- Time references outside our corpus: "Apple's 2018 R&D" (no 2018 data)
- Stock-price questions: "What's Apple's stock price?" (not in 10-K data)
""",

    "compute": """\
COMPUTE GENERATION GUIDANCE
===========================

Defining traits:
Compute queries require ARITHMETIC on retrieved values from ONE company's
filings, typically within a single fiscal year or within a single 10-K.
The system must look up values AND perform calculations to answer.

Sub-types to include in each batch:
1. Ratio queries: "What's Apple's gross margin in 2023?",
   "What's the operating margin for Microsoft?",
   "What's Tesla's R&D as a percentage of revenue?"
2. Percentage-of-total queries: "What percentage of Apple's revenue
   came from Services in 2023?", "What share of JPMorgan's revenue
   is from investment banking?"
3. Per-unit calculations: "What was Tesla's revenue per vehicle in 2024?",
   "What's UnitedHealth's revenue per member?"
4. Year-over-year (within ONE filing): "How much did Apple's R&D grow
   from 2022 to 2023?" — NOTE: this is compute because one 10-K filing
   typically shows current year + prior year comparison. Multi-year
   trend across MULTIPLE filings is research.
5. Sum/aggregate queries: "What were Walmart's total operating expenses
   in 2023?" — if the answer requires summing multiple line items
6. Derived metrics: "What's Coca-Cola's free cash flow in 2023?"
   (= operating cash flow - capex)

Variation dimensions to hit:
- Company: distribute across 20 tickers
- Calculation type: ratios, percentages, sums, differences, per-unit
- Phrasing: "What's", "Calculate", "Compute", "What percentage",
  "How much", explicit ("revenue divided by employees") vs implicit
  ("revenue per employee")
- Specificity: some queries explicit about the calculation ("revenue
  minus COGS"), others abstract ("gross margin")

Edge cases to specifically include:
- Queries that look like lookup but require arithmetic:
  "Apple's gross margin" — gross margin isn't a line item, it's calculated
- Queries with multiple values that need combining:
  "Total spent on R&D and marketing combined for Microsoft 2023"
- Within-filing year-over-year (this is the trickiest):
  "By how much did Apple's revenue grow year-over-year in 2023?"
  (the 2023 10-K shows 2023 vs 2022 figures — single filing, compute)
- Casual phrasing of computational requests:
  "what's apple's gross margin", "tesla revenue per car?"

Specifically AVOID:
- Pure lookup framed with a number: "How much was Apple's R&D in 2023?"
  (just retrieval, → lookup)
- Multi-year trends requiring multiple filings:
  "Apple's revenue growth 2020-2024" (5 filings → research)
- Multi-company comparisons: "Apple vs Microsoft gross margin"
  (→ research)
- Conceptual questions about the metric itself:
  "What is gross margin?" (→ meta)
- Investment angle: "Is Apple's gross margin healthy?" (→ decline)
- Future projection: "What will Apple's gross margin be in 2025?"
  (→ decline, prediction)
- Queries where the arithmetic is implicit but trivial:
  "Apple's revenue in millions" (just unit conversion, → lookup)
""",

    "research": """\
RESEARCH GENERATION GUIDANCE
============================

Defining traits:
Research queries require MULTI-DOCUMENT retrieval. The system cannot
answer with a single filing lookup or single-filing computation.

Any ONE of these conditions makes a query research:
- Multiple companies (2+) being compared on the same metric/dimension
- Multiple time periods spanning multiple filings (e.g., 3-year trend,
  5-year comparison)
- Multi-hop reasoning where one retrieved fact narrows the next retrieval

Sub-types to include in each batch:
1. Multi-company comparison (single year):
   "Compare Apple and Microsoft's operating margins in 2023"
   "How does Tesla's R&D spending compare to Boeing's?"
   "Which of Apple, Microsoft, Google had the highest revenue growth?"
2. Multi-period trend (single company, multiple filings):
   "How has Tesla's gross margin trended from 2022 to 2024?"
   "What's been the trajectory of Amazon's free cash flow over the
   last 3 years?"
3. Multi-company AND multi-period (the heaviest):
   "Compare Apple and Microsoft's revenue growth from 2022 to 2024"
4. Sector or category queries (implicit multi-company):
   "Which Big Tech company has the most diversified revenue?"
   "How do the major banks (JPM, BAC) compare on capital ratios?"
5. Multi-hop reasoning:
   "What was the R&D spend of the highest-revenue tech company in 2023?"
   (Step 1: identify highest-revenue tech company → Step 2: look up
   that company's R&D)
   "Which company in the corpus had the lowest gross margin in 2024?"
6. Cross-disclosure comparison: "How does Apple's risk disclosure differ
   from Microsoft's regarding AI?", "Compare Tesla's and Boeing's
   supply chain risk language"

Variation dimensions to hit:
- Number of companies compared: 2 (most common), 3 (moderate),
  4+ (rare, but include some)
- Number of time periods: 2 years (smaller research), 3-5 years
  (typical research)
- Metric breadth: financial (revenue, margins), qualitative
  (risk disclosures, MD&A language), strategic (segments, geographies)
- Phrasing: "compare", "how does X relate to Y", "trend", "trajectory",
  "evolution", "across the past N years"

Edge cases to specifically include:
- 2-year comparisons that LOOK like compute but are research:
  "How much did Apple's revenue change from 2022 to 2024?"
  (NOTE: 2022 and 2024 are in DIFFERENT 10-K filings, so this is research,
  NOT the within-filing YoY which is compute)
- Implicit multi-company: "Which of the big banks has the lowest expense
  ratio?" — multiple companies implied by "big banks"
- Qualitative multi-doc comparisons: "How does the wording of Apple's
  AI risk disclosure compare to Microsoft's?"
- Multi-hop with selection criteria: "Show me the R&D of the company
  with the highest gross margin"

Specifically AVOID:
- Within-filing YoY (the prior-year comparison shown in one 10-K):
  "How did Apple's revenue change year-over-year in 2023?" (→ compute,
  because the 2023 10-K already shows 2022 comparison)
- Open-ended "tell me everything" requests:
  "Tell me everything about Apple vs Microsoft" (too unbounded;
  → clarify in practice)
- Single-company single-year queries even if topic is broad:
  "What are Apple's risks?" (→ lookup, single filing)
- Investment-advice angle: "Which is a better buy, AAPL or MSFT?"
  (→ decline)
- Prediction angle: "Which company will grow fastest in 2025?"
  (→ decline, prediction)
- Cross-corpus references: "Compare Apple and Spotify" (Spotify not
  in corpus)
- Multiple unrelated questions bundled: "Compare Apple's R&D AND tell me
  Tesla's risks" (multi-intent; not a clean research query)
""",

    "clarify": """\
CLARIFY GENERATION GUIDANCE
===========================

Defining traits:
Clarify queries are well-formed natural-language queries that lack one
or more critical pieces of information the system MUST have to act.
The user clearly wants financial information but the query is too vague
or context-dependent to act on without asking back.

Sub-types to include in each batch:
1. Missing company: "What were earnings last quarter?",
   "How did revenue look?", "What's the gross margin?"
2. Missing time period (when the year is essential): "What's Apple's
   revenue?" (which year?), "Tell me about Microsoft's R&D" (when?)
3. Missing both: "What was operating income?", "What are the segments?"
4. Vague reference to prior context: "Tell me more about that",
   "How about the other one?", "Compare them"
   (The user is referencing something from a prior turn that we don't
   have access to in a single-turn classification)
5. Vague metric/concept: "How is Apple doing financially?",
   "Tell me about Apple", "Give me an overview of Microsoft"
   (no specific metric requested — too broad to retrieve)
6. Ambiguous time reference: "What were the latest numbers?",
   "How are things looking recently?"
   (no fixed time anchor; user could mean Q1, last year, last 3 years)

Variation dimensions to hit:
- Length: short (2-5 words: "what about earnings?") to medium
  (10-15 words: "I'm curious about how the company performed last
  quarter")
- Phrasing: question form, imperative ("tell me"), declarative with
  question mark ("revenue?")
- Vagueness type: missing entity, missing time, missing metric,
  pronoun reference, vague metric
- Tone: casual ("hows tesla doing"), neutral ("how is Tesla doing"),
  formal ("Could you provide an overview of Tesla's performance?")

Edge cases to specifically include:
- Pronoun references that are common in real conversation:
  "How did they do last year?" (who is "they"?)
  "What about the second one?" (which list?)
  "Compare them" (compare what?)
- Vague metrics that aren't quite gibberish:
  "How are Apple's numbers?" (which numbers? — clarify)
  "Tell me about Apple's situation" (situation = ?)
- Queries with company but no metric or time, where context required:
  "Apple?" — single word, but clearly a query (clarify needed)
- Questions a user MIGHT type if they forgot to mention context:
  "What's their stock price?" (whose? and 10-K doesn't have stock price
  anyway → clarify on missing company, or decline on out-of-scope)

Specifically AVOID:
- Queries that ARE answerable (with reasonable defaults):
  "What's Apple's R&D?" — defaults to most recent year → lookup, NOT clarify
- Random gibberish or malformed text: "asdf jkl revenue?"
  (clarify is for well-formed-but-underspecified, not malformed)
- Off-topic vague queries: "What time is it?" (→ decline, out_of_scope)
- Investment advice framings: "Should I invest?" (→ decline)
- Smalltalk that has no query at all: "Hi" (→ smalltalk)
- Queries that are vague but the missing info is trivially inferable:
  "what's it?" (no recoverable signal at all → don't generate;
  the directive model can't reasonably clarify this)
""",

    "decline": """\
DECLINE GENERATION GUIDANCE
===========================

Defining traits:
Decline queries are requests for things FinDirector will NOT do — either
because they cross compliance boundaries (investment advice, predictions)
or because they fall outside the system's scope (non-financial topics,
personal financial planning).

Distribute generated queries across the FOUR sub-reasons:
- investment_advice: ~35% of decline batch
- prediction: ~25% of decline batch
- out_of_scope: ~25% of decline batch
- personal_financial_advice: ~15% of decline batch

INVESTMENT_ADVICE sub-types:
1. Direct buy/sell/hold queries
2. Valuation opinions ("Is Apple overvalued?")
3. Investment quality assessments
4. Disguised-as-data investment queries (the trickiest):
   "Should I buy Apple based on their 2024 R&D spending?"
5. Portfolio queries ("Should I add Microsoft to my portfolio?")

PREDICTION sub-types:
1. Stock price predictions
2. Future performance predictions ("Will Apple's revenue grow next year?")
3. Market/macro predictions
4. Specific event predictions ("When will Apple announce earnings?")
5. Trend extrapolation ("Will Apple's margins keep expanding?")

OUT_OF_SCOPE sub-types:
1. Non-financial topics ("What's the weather?")
2. Code/technical help
3. Personal life topics
4. Financial topics outside 10-K data (stock prices, news, crypto)
5. Competitor companies not in our corpus ("What's Netflix's revenue?")
6. Different filing types ("What's in Apple's 10-Q?")

PERSONAL_FINANCIAL_ADVICE sub-types:
1. Tax planning
2. Retirement planning
3. Personal portfolio questions
4. Life-stage financial planning
5. Risk tolerance / personal advice

Edge cases to specifically include:
- Decline-disguised-as-research:
  "Compare Apple and Microsoft — which is a better buy?"
- Decline-disguised-as-compute:
  "What's Apple's PE ratio and is it cheap?"
- Out-of-scope financial queries:
  "What's Apple's current stock price?" (not in 10-K)
- Personal queries that use FinDirector data:
  "Based on Apple's R&D growth, should I buy more for my IRA?"

Specifically AVOID:
- Pure data queries with no advice/prediction angle (→ lookup)
- Vague questions that should be clarify (→ clarify)
- Smalltalk with company mention (→ smalltalk)
- Edge cases where data IS the answer:
  "What's Apple's debt-to-equity ratio?" (→ compute, computable + in scope)
""",
}


# =============================================================================
# Layer 3 — Assembly and public API
# =============================================================================

def build_generation_prompt(code: str) -> str:
    """
    Assemble the full generation prompt for a given action code.

    Combines Layer 1 (shared scaffold) with Layer 2 (per-code guidance).

    Args:
        code: The action code to generate queries for. Must be one of
            the codes in ACTION_CODES.

    Returns:
        The complete system prompt as a single string.

    Raises:
        ValueError: If `code` is not a recognized action code.
    """
    if code not in ACTION_CODES:
        raise ValueError(
            f"Unknown action code: {code!r}. "
            f"Must be one of: {ACTION_CODES}"
        )

    return BASE_INSTRUCTIONS + "\n\n" + PER_CODE_GUIDANCE[code]


def _strip_code_fences(text: str) -> str:
    """
    Strip markdown code fences from a response if present.

    Same defensive parsing pattern as the labeler.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text

    first_newline = text.find("\n")
    if first_newline != -1:
        text = text[first_newline + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text


def generate_queries(
    client: Anthropic,
    code: str,
    n: int,
    model: str = "claude-sonnet-4-5",
) -> tuple[list[str], dict[str, int]]:
    """
    Generate N synthetic queries for the given action code.

    Args:
        client: An Anthropic API client.
        code: The action code to generate queries for.
        n: Number of queries to generate in this batch.
        model: The Claude model to use.

    Returns:
        A tuple of (queries, usage):
        - queries: list of N generated query strings
        - usage: dict with input_tokens and output_tokens

    Raises:
        ValueError: If `code` is not recognized or response is malformed.
    """
    system_prompt = build_generation_prompt(code)

    user_message = (
        f"Generate {n} diverse, realistic queries that match the "
        f"`{code}` action code. Follow all the guidance above. "
        f"Output a JSON array of {n} query strings."
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = response.content[0].text
    cleaned = _strip_code_fences(response_text)

    try:
        queries = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse generation response as JSON: {e}. "
            f"Raw response (first 500 chars): {response_text[:500]}"
        )

    if not isinstance(queries, list):
        raise ValueError(
            f"Expected JSON array, got {type(queries).__name__}: "
            f"{queries!r}"
        )

    if not all(isinstance(q, str) for q in queries):
        raise ValueError(
            f"Expected array of strings, found non-string element(s)"
        )

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return queries, usage
