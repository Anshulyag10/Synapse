"""Typed data models for every pipeline stage.

Every inter-component boundary in Synapse passes one of these models
rather than an untyped dictionary.  This makes the evidence flow explicit
and self-documenting.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────

class QueryCategory(str, Enum):
    """Categories assigned by the query router."""
    DIRECT_LOOKUP = "direct_lookup"
    SEMANTIC_SEARCH = "semantic_search"
    RELATIONAL_QUERY = "relational_query"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"


class SourceType(str, Enum):
    """Where a piece of evidence originated."""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR_INDEX = "vector_index"
    LEXICAL_INDEX = "lexical_index"


class ClaimStatus(str, Enum):
    """Grounding status of a single factual claim."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


# ── Query Planning ─────────────────────────────────────────────────────

class QueryPlan(BaseModel):
    """Structured output from the retrieval planner."""
    original_query: str
    query_type: QueryCategory
    sources: list[SourceType]
    requires_reranking: bool = True
    requires_verification: bool = True
    reasoning: str = Field("", description="Brief explanation of why this plan was chosen")


class RetrievalRequest(BaseModel):
    """A request sent to one of the retrieval backends."""
    query: str
    source_type: SourceType
    top_k: int = 5
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── Evidence Objects ───────────────────────────────────────────────────

class RetrievalCandidate(BaseModel):
    """A single item returned by any retriever, before fusion."""
    candidate_id: str = ""
    text: str
    score: float = 0.0
    rank: int = 0
    source_type: SourceType
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: str = Field("", description="Human-readable origin, e.g. 'Neo4j entity:fever'")


class GraphEvidence(BaseModel):
    """Evidence extracted from a knowledge graph traversal."""
    entity_name: str
    entity_type: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    traversal_depth: int = 0
    provenance: str = ""


class VectorEvidence(BaseModel):
    """Evidence from the dense vector index."""
    document_id: str
    text: str
    similarity_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: str = ""


class LexicalEvidence(BaseModel):
    """Evidence from the BM25 lexical index."""
    document_id: str
    text: str
    bm25_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: str = ""


class EvidenceBundle(BaseModel):
    """All evidence collected for a single query, after fusion."""
    graph_evidence: list[GraphEvidence] = Field(default_factory=list)
    vector_evidence: list[VectorEvidence] = Field(default_factory=list)
    lexical_evidence: list[LexicalEvidence] = Field(default_factory=list)
    fused_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    source_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of items per source type",
    )


# ── Grounding & Citations ─────────────────────────────────────────────

class Citation(BaseModel):
    """Maps a single claim in the answer to its supporting evidence."""
    claim_text: str
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="IDs or short descriptions of evidence items",
    )
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class GroundingResult(BaseModel):
    """Output of the evidence verifier."""
    supported_claims: list[Citation] = Field(default_factory=list)
    unsupported_claims: list[Citation] = Field(default_factory=list)
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    total_claims: int = 0
    grounded_ratio: float = Field(0.0, ge=0.0, le=1.0)


# ── Agent Decisions ────────────────────────────────────────────────────

class AgentDecision(BaseModel):
    """A single decision/action taken by the orchestration agent."""
    action: str
    reasoning: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    latency_ms: float = 0.0


class AnswerTrace(BaseModel):
    """Complete provenance trace for one query → answer cycle."""
    query: str
    query_plan: Optional[QueryPlan] = None
    decisions: list[AgentDecision] = Field(default_factory=list)
    evidence_bundle: Optional[EvidenceBundle] = None
    grounding_result: Optional[GroundingResult] = None
    citations: list[Citation] = Field(default_factory=list)
    answer: str = ""
    evidence_confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Interpretable confidence signal (NOT a calibrated probability)",
    )
    retrieval_strategy: str = ""
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# ── Evaluation ─────────────────────────────────────────────────────────

class EvaluationRecord(BaseModel):
    """Per-query evaluation result."""
    query_id: int | str = ""
    category: str = ""
    subtype: str = ""
    difficulty: str = ""
    retrieval_type: str = ""
    question: str = ""
    gold_names: list[str] = Field(default_factory=list)
    ground_truth: str = ""
    answer: str = ""
    retrieved_top_k: list[str] = Field(default_factory=list)
    rank: Optional[int] = None
    gold_found: bool = False
    recall_at_1: int = 0
    recall_at_3: int = 0
    recall_at_5: int = 0
    reciprocal_rank: float = 0.0
    ndcg_at_5: float = 0.0
    routing_recall: Optional[float] = None
    evidence_recall: Optional[float] = None
    citation_coverage: Optional[float] = None
    judge_score: Optional[float] = None
    judge_verdict: str = ""
    latency_ms: dict[str, float] = Field(default_factory=dict)
    tools_called: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)
