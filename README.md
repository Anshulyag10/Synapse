# EvidenceGraph — Neuro-Symbolic Agentic RAG for Evidence-Grounded Question Answering

I built EvidenceGraph as a neuro-symbolic retrieval-augmented generation system that
combines knowledge graph traversal, dense vector retrieval, BM25 lexical search,
cross-encoder reranking, and LLM-based synthesis to produce evidence-grounded
answers with provenance tracking.

Healthcare is the current **demonstration domain** — the architecture is designed to
be domain-agnostic, with healthcare-specific mappings isolated in a domain adapter.

> **Educational/Research Use Only**: This is not a clinical tool.

## Architecture

```
                              ┌───────────────────┐
                              │   User Query       │
                              └────────┬──────────┘
                                       │
                              ┌────────▼──────────┐
                              │   Query Router     │
                              │ (LLM + rule-based) │
                              └────────┬──────────┘
                                       │  QueryCategory
                              ┌────────▼──────────┐
                              │ Retrieval Planner  │
                              │   → QueryPlan      │
                              └────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
           ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼───────┐
           │ Knowledge    │  │   Vector     │  │   Lexical     │
           │ Graph (Neo4j)│  │   (ChromaDB) │  │   (BM25)      │
           └───────┬──────┘  └───────┬──────┘  └───────┬───────┘
                    │                  │                   │
                    └──────────┬───────┴───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Evidence Fusion    │
                    │  (normalise, dedup, │
                    │   weighted rank)    │
                    └──────────┬──────────┘
                               │  EvidenceBundle
                    ┌──────────▼──────────┐
                    │  Synthesis Agent    │
                    │  (LLM generation)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Grounding Verifier  │
                    │ (claim→evidence map)│
                    └──────────┬──────────┘
                               │  AnswerTrace
                    ┌──────────▼──────────┐
                    │   Grounded Answer   │
                    │   with citations    │
                    └─────────────────────┘
```

## Design Decisions

**Why hybrid retrieval?**
- The knowledge graph gives explicit relational reasoning (symptom → disease → treatment).
- Dense vector retrieval gives semantic search over unstructured descriptions.
- BM25 lexical retrieval catches exact terminology that embeddings dilute.
- The agent decides per-query which sources to activate via structured planning.

**Why a query router?**
- Different query types benefit from different retrieval strategies.
- LLM classification with rule-based fallback keeps the system responsive when the
  LLM is unavailable or rate-limited.

**Why evidence fusion?**
- Multi-source results need normalised scoring and deduplication before ranking.
- Weighted fusion lets us tune the contribution of each source type.

**Why grounding verification?**
- Best-effort check that generated claims are supported by retrieved evidence.
- Not perfect hallucination prevention — documented as a signal, not a guarantee.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Knowledge Graph | Neo4j | Entity-relationship store (diseases, symptoms, treatments) |
| Vector Index | ChromaDB | Dense semantic search over document corpus |
| Lexical Index | BM25 (rank-bm25) | Keyword retrieval for exact terminology |
| Embeddings | BAAI/bge-base-en-v1.5 | 768-dim dense vectors |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder for candidate reranking |
| Agent LLM | Groq (llama-3.3-70b-versatile) | Tool-calling orchestration agent |
| Normalisation LLM | Ollama (qwen2.5:7b-instruct) | Local model for KG ingestion and benchmarks |
| API | FastAPI + Uvicorn | REST service |
| Frontend | Streamlit | Interactive chat UI |
| Config | Pydantic BaseSettings | Centralised typed configuration |
| Data Models | Pydantic BaseModel | Typed pipeline objects |

## Installation

```bash
git clone <repo-url>
cd EvidenceGraph
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

**Prerequisites:**
- Python 3.10+
- Groq API key
- Neo4j (local or [Aura](https://neo4j.com/cloud/aura/))
- Ollama with `qwen2.5:7b-instruct` pulled (KG ingestion only)
- CUDA GPU recommended for embedding and reranking

Create `.env` from the example:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Data Sources

**Entity Data** (`data/healthcare/disease_data.csv`) — included
- Disease names, symptoms, treatments, contagious/chronic flags

**Document Corpus** — download separately
- Source: [Indian Medicine Dataset (Kaggle)](https://www.kaggle.com/datasets/mohneesh7/indian-medicine-data)
- ~195k medicine records with compositions, side effects, interactions
- Place in `data/healthcare/medicine_data.csv`

> Data from Kaggle — not clinically validated.

## Data Ingestion

**1. Knowledge Graph (Neo4j)** — requires Ollama running
```bash
ollama pull qwen2.5:7b-instruct
python run.py ingest-graph
```
Normalises entities via LLM, resolves synonyms, writes the graph.

**2. Vector + Lexical Index (ChromaDB)**
```bash
python run.py ingest-vector
```
Cleans CSV → `corpus_records.json` → embeds → ChromaDB. Takes ~2-3 hours on GPU.

## Usage

### Interactive CLI
```bash
python run.py                  # or: python run.py query
```
Type `reset` to clear conversation, `exit` to quit.

### FastAPI Service
```bash
python run.py serve
# → http://localhost:8000/docs (Swagger UI)
```

Endpoints:
- `POST /query` — full evidence-grounded QA
- `POST /retrieve` — retrieval only
- `GET /health` — service health
- `GET /config` — current configuration

### Streamlit Frontend
```bash
python run.py streamlit
```

## Evaluation

I built a benchmark and scoring harness to measure retrieval, routing,
and answer quality.

```bash
python run.py benchmark               # generate Evaluation/benchmark.csv
python -m evidence_graph.evaluation    # run evaluation
```

**Metrics measured:**
- **Retrieval**: Recall@1/3/5, MRR, nDCG@5
- **Routing**: routing accuracy, expected-tool recall
- **Answer quality**: evidence recall, gold mention, LLM-as-judge
- **Grounding**: citation coverage, grounded claim ratio

### Ablation Framework

Seven configurations to measure component contributions:
- A. Vector only → B. KG only → C. BM25 only → D. Vector+BM25
- E. KG+Vector → F. KG+Vector+BM25 → G. Full agentic system

## Project Structure

```
EvidenceGraph/
├── src/evidence_graph/
│   ├── core/              # Config, models, exceptions, logging
│   ├── retrieval/          # Graph, vector, lexical, hybrid retrievers
│   ├── agents/             # Router, planner, evidence agent, synthesis
│   ├── grounding/          # Evidence fusion, verifier, citations
│   ├── ingestion/          # Graph builder, vector indexer, normalisation
│   ├── evaluation/         # Metrics, benchmark, ablation framework
│   ├── domain/             # Healthcare adapter (swappable)
│   └── application/        # FastAPI, CLI, Streamlit
├── tests/                  # Pytest test suite
├── data/healthcare/        # Source CSV data
├── docs/                   # Architecture and design documentation
├── run.py                  # Convenience entry point
├── pyproject.toml          # Project configuration
└── .env.example            # Environment template
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Limitations

- Data from Kaggle CSVs — not clinically validated
- Not for real diagnosis or treatment decisions
- No personalisation (age, allergies, patient history)
- LLM responses may vary between runs
- Grounding verification is best-effort, not a hallucination guarantee

## License

MIT. See individual dependencies for their licenses.
