# Agent Architecture

## Overview

EvidenceGraph uses a layered agent architecture:

```
Query → Router → Planner → Evidence Agent (LLM loop) → Synthesis → Verification
```

Each layer has a distinct responsibility and produces typed output.

## Query Router

**File:** `agents/router.py`

Classifies queries into 5 categories:

| Category | Description | Example |
|----------|-------------|---------|
| `direct_lookup` | Named-entity query | "What are the side effects of aspirin?" |
| `semantic_search` | Descriptive query | "medicines for headache" |
| `relational_query` | Relationship query | "does aspirin interact with ibuprofen?" |
| `multi_hop` | Chained reasoning | "I have fever and cough, what medicine should I take?" |
| `ambiguous` | Unclear query | "help me" |

### Implementation
- **Primary:** LLM-based classification with structured JSON output
- **Fallback:** keyword-pattern heuristics
- Routing decisions are logged for evaluation

## Retrieval Planner

**File:** `agents/planner.py`

Converts a QueryCategory into a structured QueryPlan:

| Category | Sources Activated | Rerank | Verify |
|----------|------------------|--------|--------|
| `direct_lookup` | KG + Vector | No | No |
| `semantic_search` | Vector + Lexical | Yes | Yes |
| `relational_query` | KG | No | Yes |
| `multi_hop` | KG + Vector + Lexical | Yes | Yes |
| `ambiguous` | All | Yes | Yes |

## Evidence Agent

**File:** `agents/evidence_agent.py`

The central orchestrator with an LLM tool-calling loop:

1. Receives user query
2. Classifies via router
3. Plans via planner
4. Runs the LLM tool-calling loop (up to 6 rounds)
5. Dispatches tool calls to retrieval backends
6. Returns AnswerTrace with full provenance

### Tools Available

| Tool | Backend | Purpose |
|------|---------|---------|
| `retrieve_entities_by_attributes` | Graph | Find entities by attributes |
| `get_entity_details` | Graph | Full entity record |
| `get_document_details` | Vector | Exact document lookup |
| `search_corpus` | Vector | Semantic search |

### Error Recovery

The agent recovers from malformed LLM tool calls by:
1. Parsing the failed generation for tool name and arguments
2. Dispatching the recovered call directly
3. Falling back to synthesis-without-tools if recovery fails

## Synthesis Agent

**File:** `agents/synthesis_agent.py`

Generates answers from evidence bundles and computes confidence:
- Evidence confidence score based on source diversity, volume, and cross-method agreement
- NOT a calibrated probability — an interpretable signal
