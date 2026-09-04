# Evaluation Framework

## Overview

EvidenceGraph includes a benchmark generator, scoring harness, and ablation
framework for measuring retrieval, routing, and answer quality.

## Metrics

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| Recall@1/3/5 | Fraction of queries where the gold item appears in top-k |
| MRR | Mean Reciprocal Rank — average of 1/rank |
| nDCG@5 | Normalised Discounted Cumulative Gain at 5 |
| Latency | Retrieval time in milliseconds (mean, p50, p95) |

### Routing Metrics
| Metric | Description |
|--------|-------------|
| Routing accuracy | Fraction of correctly classified queries |
| Expected-tool recall | Fraction of expected tools actually called |
| Unnecessary-call rate | Fraction of called tools not in the expected set |

### Generation Metrics
| Metric | Description |
|--------|-------------|
| Evidence recall | Fraction of gold evidence items in the answer |
| Gold mention | Whether the gold entity name appears in the answer |
| LLM-as-judge | Automated grading (correct / partial / incorrect) |
| Citation coverage | Fraction of claims with supporting citations |

## Benchmark

The benchmark generator creates labeled questions from source CSVs:

- **Entity by attributes** (semantic) — describes attributes without naming the entity
- **Entity by name** (known-item) — names the entity explicitly
- **Document by name** (known-item) — exact document lookup
- **Document by indication** (semantic) — describe an indication
- **Document by side-effect** (semantic) — describe a side effect
- **Multi-hop** — chained retrieval (symptoms → entity → treatment)

### Integrity Guarantees
- Semantic questions never leak the answer name (fuzz guard)
- Known-item questions name the target deliberately
- Every row carries: ground_truth, expected_tools, gold accept-set,
  difficulty, retrieval_type, score_mode

## Ablation Study

Seven configurations measuring component contributions:

| Config | Sources | Purpose |
|--------|---------|---------|
| A | Vector only | Dense retrieval baseline |
| B | KG only | Graph retrieval baseline |
| C | BM25 only | Lexical retrieval baseline |
| D | Vector + BM25 | Dense + lexical |
| E | KG + Vector | Graph + dense |
| F | KG + Vector + BM25 | All retrievers, no routing |
| G | Full agentic | Complete system with routing |

Results are generated from actual evaluation runs — never fabricated.
