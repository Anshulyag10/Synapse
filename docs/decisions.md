# Architectural Decision Records

## ADR-1: Why Neo4j for the Knowledge Graph?

**Context:** The system needs structured entity-relationship storage for
disease-symptom-treatment triples with traversal queries.

**Decision:** Use Neo4j with Cypher.

**Rationale:**
- Native graph traversal makes multi-hop queries (symptom → disease → treatment)
  expressible as single Cypher statements
- UNIQUE constraints enforce entity deduplication at the database level
- Free tier (Aura) sufficient for the demonstration dataset (~400 diseases)
- Well-supported Python driver

**Alternatives considered:**
- NetworkX (in-memory) — would not persist across restarts
- PostgreSQL with recursive CTEs — possible but more complex for graph patterns

---

## ADR-2: Why ChromaDB for the Vector Index?

**Context:** Need a persistent vector store for ~195k medicine record embeddings
with ANN search.

**Decision:** Use ChromaDB with persistent storage.

**Rationale:**
- Zero-configuration local deployment (no external service)
- Built-in persistence to disk
- HNSW index with cosine similarity
- Metadata filtering support

**Alternatives considered:**
- FAISS — faster but no built-in persistence or metadata filtering
- Pinecone — managed service, adds external dependency
- Weaviate — heavier deployment

---

## ADR-3: Why BM25 as a Third Retrieval Backend?

**Context:** Dense embeddings may dilute exact terminology (drug names,
compositions) that users type verbatim.

**Decision:** Add BM25 lexical retrieval as a complement to dense retrieval.

**Rationale:**
- Catches exact drug names and compositions that BGE embeddings average out
- No additional infrastructure — built in-memory from the same corpus
- Measurable via ablation (config C vs D vs F)

---

## ADR-4: Why Cross-Encoder Reranking?

**Context:** Initial retrieval (ANN or candidate generation) returns a large
pool of candidates. Ranking precision matters for the top-k.

**Decision:** Use BAAI/bge-reranker-v2-m3 cross-encoder shared across all
retrieval backends.

**Rationale:**
- Cross-encoders score (query, passage) pairs jointly, achieving higher
  precision than bi-encoder similarity alone
- A single shared instance avoids redundant GPU memory
- The model is small enough to run on consumer GPUs

---

## ADR-5: Why Agentic Tool-Calling Instead of a Fixed Pipeline?

**Context:** Different query types (symptoms, named lookups, interactions,
multi-hop) benefit from different retrieval strategies.

**Decision:** Use an LLM tool-calling agent with structured routing.

**Rationale:**
- The LLM can chain tools (symptoms → diseases → medicines) for multi-hop queries
- An explicit query router provides the first level of classification
- The LLM decides tool ordering within its round budget
- Malformed tool-call recovery preserves robustness

**Trade-off:** LLM latency per query (mitigated by Groq's fast inference).

---

## ADR-6: Why Evidence Fusion with Weighted Scoring?

**Context:** Multi-source retrieval produces results with incomparable
score scales (BM25 scores, cosine similarities, graph match counts).

**Decision:** Min-max normalise per source, then apply configurable weights.

**Rationale:**
- Simple and interpretable
- Weights are configurable for domain-specific tuning
- Ablation framework can measure the effect of different weight configurations

---

## ADR-7: Why Best-Effort Grounding Verification?

**Context:** LLM-generated answers may include hallucinated claims.

**Decision:** Extract claims, check keyword overlap with evidence, classify
as supported/partially/unsupported.

**Rationale:**
- Provides an interpretable signal without requiring a second LLM call
- Keyword overlap is fast and deterministic
- Documented as best-effort — not a hallucination guarantee
- Can be upgraded to semantic similarity or LLM-based NLI in future
