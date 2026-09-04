"""FastAPI service for EvidenceGraph.

Endpoints:
  POST /query     — full evidence-grounded QA
  POST /retrieve  — retrieval only (no LLM synthesis)
  GET  /health    — service health check
  GET  /config    — current configuration (secrets redacted)
"""

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evidence_graph.core.config import EvidenceGraphSettings, load_settings
from evidence_graph.core.logging import get_logger, setup_logging

logger = get_logger("application.api")

app = FastAPI(
    title="EvidenceGraph",
    description=(
        "Neuro-symbolic agentic RAG for evidence-grounded question answering. "
        "Healthcare is the current demonstration domain."
    ),
    version="1.0.0",
)

# ── Request / Response models ──────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str = Field(..., description="Natural-language question")
    include_trace: bool = Field(False, description="Include full provenance trace")


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""
    answer: str
    retrieval_strategy: str = ""
    evidence_confidence: float = 0.0
    citations: list[dict] = Field(default_factory=list)
    trace: Optional[dict] = None


class RetrieveRequest(BaseModel):
    """Request body for the /retrieve endpoint."""
    query: str
    sources: list[str] = Field(
        default=["knowledge_graph", "vector_index", "lexical_index"],
        description="Which retrieval backends to activate",
    )


class RetrieveResponse(BaseModel):
    """Response body for the /retrieve endpoint."""
    graph_results: dict = Field(default_factory=dict)
    vector_results: dict = Field(default_factory=dict)
    lexical_results: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    graph_connected: bool
    vector_connected: bool
    lexical_ready: bool


# ── Lazy-loaded engine ─────────────────────────────────────────────────

_engine = None


def _get_engine():
    """Lazily initialise the EvidenceGraph engine."""
    global _engine
    if _engine is None:
        from evidence_graph.retrieval.graph_retriever import GraphRetriever
        from evidence_graph.retrieval.vector_retriever import VectorRetriever
        from evidence_graph.agents.evidence_agent import EvidenceAgent

        cfg = load_settings()
        setup_logging(cfg.log_level, cfg.log_dir)
        graph = GraphRetriever(cfg)
        vector = VectorRetriever(cfg)
        _engine = {
            "agent": EvidenceAgent(graph, vector, cfg),
            "graph": graph,
            "vector": vector,
            "cfg": cfg,
        }
    return _engine


# ── Endpoints ──────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Full evidence-grounded question answering."""
    engine = _get_engine()
    try:
        trace = engine["agent"].query(request.query)
        response = QueryResponse(
            answer=trace.answer,
            retrieval_strategy=trace.retrieval_strategy,
            evidence_confidence=trace.evidence_confidence,
        )
        if request.include_trace:
            response.trace = trace.model_dump()
        return response
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(request: RetrieveRequest):
    """Retrieval only — no LLM synthesis."""
    engine = _get_engine()
    result = RetrieveResponse()
    try:
        if "knowledge_graph" in request.sources:
            r = engine["graph"].retrieve_by_attributes([request.query])
            result.graph_results = r
        if "vector_index" in request.sources:
            r = engine["vector"].retrieve_by_query(request.query)
            result.vector_results = r
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Service health check."""
    engine = _get_engine()
    return HealthResponse(
        status="ok",
        graph_connected=engine["graph"].connected,
        vector_connected=engine["vector"].connected,
        lexical_ready=True,
    )


@app.get("/config")
async def config_endpoint():
    """Current configuration (secrets redacted)."""
    cfg = load_settings()
    d = cfg.model_dump()
    # Redact secrets
    for key in ("groq_api_key", "neo4j_password"):
        if d.get(key):
            d[key] = "***"
    return d
