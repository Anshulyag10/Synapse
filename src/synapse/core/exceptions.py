"""Custom exception hierarchy for Synapse.

Every major subsystem raises a specific exception so callers can degrade
gracefully — e.g. if the knowledge graph is offline, the system can still
answer from the vector and lexical indexes.
"""


class SynapseError(Exception):
    """Base for all Synapse exceptions."""


# ── Retrieval ──────────────────────────────────────────────────────────

class RetrievalError(SynapseError):
    """A retrieval backend encountered an unrecoverable error."""


class GraphConnectionError(RetrievalError):
    """Neo4j is unreachable or returned a driver-level error."""


class VectorDBError(RetrievalError):
    """ChromaDB is unreachable or a query against it failed."""


class LexicalIndexError(RetrievalError):
    """The BM25 index is missing or corrupt."""


class EmptyRetrievalError(RetrievalError):
    """All retrieval paths returned zero candidates for a query."""


# ── Ingestion ──────────────────────────────────────────────────────────

class IngestionError(SynapseError):
    """A data-ingestion pipeline failed."""


class NormalizationError(IngestionError):
    """The LLM-based entity normalizer produced unusable output."""


class DataSourceError(IngestionError):
    """A required data source (CSV, JSON) is missing or unreadable."""


# ── Agent / LLM ────────────────────────────────────────────────────────

class AgentError(SynapseError):
    """The orchestration agent encountered an unrecoverable problem."""


class MalformedToolCallError(AgentError):
    """The LLM emitted a tool call that could not be parsed."""


class LLMTimeoutError(AgentError):
    """An LLM API call exceeded its timeout budget."""


# ── Grounding ──────────────────────────────────────────────────────────

class GroundingError(SynapseError):
    """Evidence verification or citation mapping failed."""


class InsufficientEvidenceError(GroundingError):
    """Not enough evidence was retrieved to ground the answer."""


# ── Configuration ──────────────────────────────────────────────────────

class ConfigurationError(SynapseError):
    """A required configuration value is missing or invalid."""
