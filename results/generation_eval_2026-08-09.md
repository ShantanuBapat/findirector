# Generation Eval — 2026-08-09

- eval set: `data/eval/eval_set.jsonl` (40 cases)
- **overall: 70.0%** (28/40)
- numeric: 63.6% (22); text: 77.8% (18)
- judge: claude-sonnet-4-5; elapsed 414.2s

| id | ticker | type | correct | expected | actual |
|----|--------|------|---------|----------|--------|
| eval_001 | BAC | text | ✓ | Bank of America is subject to the CCPA ( | Based on the provided SEC 10-K excerpts, |
| eval_002 | WMT | numeric | ✓ | $39,579.0 million | Based on the SEC 10-K excerpts provided, |
| eval_003 | TSLA | numeric | ✓ | Approximately $245 million | According to Tesla's 10-K filing, **the  |
| eval_004 | MSFT | numeric | ✓ | $22 million | Based on the SEC 10-K filing excerpts pr |
| eval_005 | BRK-B | text | ✗ | The reclassifications were associated wi | Based on the provided excerpts, I cannot |
| eval_006 | TSLA | numeric | ✗ | $832.0 million | Based on the SEC 10-K excerpts provided: |
| eval_007 | PFE | text | ✓ | The third-party matrix-pricing model use | According to Pfizer's SEC 10-K filings,  |
| eval_008 | XOM | text | ✓ | ExxonMobil faces competition from other  | Based on the SEC 10-K excerpts provided, |
| eval_009 | AAPL | text | ✓ | Apple historically experiences higher ne | Based on the SEC 10-K excerpts provided, |
| eval_010 | XOM | text | ✓ | A material decline in oil or natural gas | Based on the provided SEC 10-K excerpts, |
| eval_011 | JPM | text | ✗ | Quality and variety of products and serv | I don't have SEC filings for JPM in my c |
| eval_012 | WMT | text | ✓ | The Health Insurance Portability and Acc | According to the SEC 10-K excerpts, Walm |
| eval_013 | PFE | text | ✗ | July 2023 | I cannot find information about when Pfi |
| eval_014 | V | text | ✓ | Visa's portfolio of commercial payments  | Based on the SEC 10-K excerpts provided, |
| eval_015 | JNJ | numeric | ✗ | $789 million | Based on the SEC 10-K excerpts provided, |
| eval_016 | GOOGL | numeric | ✓ | $3,865 million | According to Alphabet's SEC 10-K filing, |
| eval_017 | CVX | numeric | ✓ | 3.3 million barrels per day | According to the SEC 10-K filing, Chevro |
| eval_018 | PG | text | ✓ | 1) changes in the cash flows or other ma | According to the SEC 10-K filings, the t |
| eval_019 | AMZN | numeric | ✓ | $82,312.0 million | According to Amazon's SEC 10-K filing, t |
| eval_020 | NVDA | text | ✓ | No, NVDA has not entered into any financ | Based on the provided excerpts, **NVDA h |
| eval_021 | NVDA | numeric | ✗ | Over 78% of the supercomputers on the gl | That's outside what I do. I answer quest |
| eval_022 | META | numeric | ✗ | $14,681.0 million | I don't have SEC filings for META in my  |
| eval_023 | KO | numeric | ✗ | $(14,131) million | I don't have SEC filings for KO in my co |
| eval_024 | META | numeric | ✓ | $1,639 million | According to Meta's fiscal year 2023 10- |
| eval_025 | BA | numeric | ✗ | 99% | That question needs a capability I don't |
| eval_026 | AMZN | numeric | ✓ | $11.8 billion | According to Amazon's SEC 10-K filing, * |
| eval_027 | UNH | text | ✓ | State net operating loss carryforwards e | According to the provided SEC 10-K excer |
| eval_028 | KO | text | ✓ | Less than 20 years | Based on the SEC 10-K excerpts provided, |
| eval_029 | JNJ | text | ✓ | The service cost component is presented  | According to JNJ's SEC 10-K filings, the |
| eval_030 | BA | numeric | ✓ | $4,550 million | According to Boeing's SEC 10-K filing, t |
| eval_031 | PG | numeric | ✗ | 15% | That's outside what I do. I answer quest |
| eval_032 | JPM | text | ✓ | Financial technology companies may not b | Based on the SEC 10-K excerpts, JPMorgan |
| eval_033 | MSFT | numeric | ✓ | $88,136 million | Based on the SEC 10-K filing excerpts pr |
| eval_034 | V | numeric | ✓ | $5.1 billion | According to the SEC 10-K filing, Visa's |
| eval_035 | UNH | numeric | ✓ | $24,365.0 million | Based on the provided SEC 10-K excerpts, |
| eval_036 | GOOGL | numeric | ✗ | $879 million | Which company's allowance for credit los |
| eval_037 | BRK-B | numeric | ✓ | $271,588 million | Based on the SEC 10-K excerpts provided, |
| eval_038 | CVX | numeric | ✓ | 3 | Based on the provided SEC 10-K excerpts, |
| eval_039 | BAC | text | ✗ | China could retaliate through the use of | I don't have SEC filings for BAC in my c |
| eval_040 | AAPL | text | ✓ | Apple has a minority market share in the | Based on the SEC 10-K excerpts provided, |
