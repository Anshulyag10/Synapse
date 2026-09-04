"""Structured logging for Synapse.

Every query produces a trace that flows through the pipeline:
  query → route → retrieval → candidates → reranking → evidence → answer
        → verification → final answer

This module provides a per-query trace logger that makes debugging possible
without reading scattered print statements.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional

# Per-request trace ID so log lines from the same query can be correlated.
_trace_id: ContextVar[str] = ContextVar("trace_id", default="no-trace")


def new_trace_id() -> str:
    """Generate and set a fresh trace ID for the current async/thread context."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """Return the current context's trace ID."""
    return _trace_id.get()


class _TraceFormatter(logging.Formatter):
    """Injects the current trace ID into every log record."""

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = _trace_id.get()  # type: ignore[attr-defined]
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    component: str = "synapse",
) -> logging.Logger:
    """Configure the root Synapse logger with console + optional file output.

    Parameters
    ----------
    level
        Minimum severity for the console handler.  The file handler always
        logs at DEBUG.
    log_dir
        If given, a timestamped log file is created in this directory.
    component
        Logger name (default ``synapse``).

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(component)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = _TraceFormatter(
        "%(asctime)s [%(trace_id)s] %(levelname)-7s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — respects the requested level.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — always DEBUG for post-mortem analysis.
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            log_path / f"{component}_{stamp}.log", encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "synapse") -> logging.Logger:
    """Return a child logger (creates the root logger lazily if needed)."""
    root = logging.getLogger("synapse")
    if not root.handlers:
        setup_logging()
    if name == "synapse":
        return root
    return root.getChild(name)
