# Retrieval Strategies

## Overview

EvidenceGraph uses three complementary retrieval backends, each with different strengths:

| Backend | Strength | Weakness | When Used |
|---------|----------|----------|-----------|
| Knowledge Graph | Explicit relationships, structured reasoning | Limited to ingested entity schema | Symptom→disease, entity details, relational queries |
| Vector Index | Semantic similarity, handles paraphrasing | May dilute exact terms | Description matching, indication search |
| Lexical (BM25) | Exact terminology, fast | No semantic understanding | Drug names, compositions, specific terms |

## Graph Retriever

**Pipeline:** user terms → exact/alias/fuzzy/BM25 candidates → cross-encoder reranking → Cypher graph expansion

### Candidate Generation Cascade

1. **Exact match** — canonical entity name lookup (O(1) in hash map)
2. **Alias match** — check against registered aliases
3. **Fuzzy match** — token_sort_ratio ≥ threshold (default 80)
4. **BM25 fallback** — keyword search over all entity surface forms

### Graph Expansion

After selecting seed nodes via cross-encoder reranking:
- 1-hop: collect all connected attributes (symptoms, treatments)
- 2-hop (optional): find related entities sharing treatment edges

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fuzzy_match_cutoff` | 80 | Minimum RapidFuzz score |
| `kg_seed_count` | 5 | Seeds after reranking |
| `max_graph_candidates` | 25 | Candidate pool before reranking |
| `max_expansion_nodes` | 5 | Max entities returned |
| `max_hops` | 2 | Traversal depth |

## Vector Retriever

**Pipeline:** query → BGE embedding → ChromaDB ANN → cross-encoder reranking → top-k

### Three Retrieval Paths

1. **Exact name** — direct lookup in the in-memory name index
2. **Composition match** — match against the composition index
3. **Semantic search** — full embed→ANN→rerank pipeline

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | BAAI/bge-base-en-v1.5 | 768-dim embeddings |
| `candidate_pool_size` | 60 | ANN candidates before reranking |
| `top_k_retrieval` | 5 | Final results after reranking |
| `reranker_model` | BAAI/bge-reranker-v2-m3 | Cross-encoder |

## Lexical Retriever (BM25)

**Pipeline:** query → tokenize → BM25 score → top-k

Built over the same `corpus_records.json` as the vector index.  Searchable text
includes: product name, composition, description, and side effects.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bm25_top_k` | 20 | Top-k from BM25 |

## Evidence Fusion

After all backends return results:
1. **Normalise scores** — min-max per source type
2. **Weight** — apply source weights (default: graph=0.35, vector=0.40, lexical=0.25)
3. **Deduplicate** — merge by name, keep highest score
4. **Rank** — sort by weighted score, assign ranks

### Fusion Weights

| Source | Default Weight | Rationale |
|--------|---------------|-----------|
| Knowledge Graph | 0.35 | Explicit relationships, high precision |
| Vector Index | 0.40 | Semantic coverage, handles paraphrasing |
| Lexical Index | 0.25 | Complementary exact matching |
