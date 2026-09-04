"""CLI interface for Synapse.

Subcommands:
  query    — interactive QA REPL
  ingest   — run data ingestion pipelines
  evaluate — run the evaluation suite
  serve    — start the FastAPI service
"""

import sys

from synapse.core.config import load_settings
from synapse.core.logging import setup_logging, get_logger

# Keep stdout/stderr UTF-8 safe
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


logger = get_logger("application.cli")


class SynapseEngine:
    """Wires the retrieval backends and agent into a single QA engine."""

    def __init__(self):
        from synapse.retrieval.graph_retriever import GraphRetriever
        from synapse.retrieval.vector_retriever import VectorRetriever
        from synapse.agents.evidence_agent import EvidenceAgent

        cfg = load_settings()
        setup_logging(cfg.log_level, cfg.log_dir)

        print("\n" + "*" * 80)
        print("  Initialising Synapse")
        print("*" * 80)

        self._graph = GraphRetriever(cfg)
        self._vector = VectorRetriever(cfg)
        self._agent = EvidenceAgent(self._graph, self._vector, cfg)

        print("\n  Ready!\n")

    def ask(self, query: str) -> str:
        """Answer a single query through the evidence agent."""
        return self._agent.chat(query)

    def reset(self):
        """Start a new conversation."""
        self._agent.reset()


def run_interactive(engine: SynapseEngine):
    """Run the interactive REPL loop."""
    print("\n" + "*" * 80)
    print("  Interactive Mode — 'exit' to quit, 'reset' to start a new conversation")
    print("*" * 80 + "\n")

    while True:
        try:
            query = input("  You: ").strip()
            if query.lower() in ("exit", "quit", "q"):
                print("\n  Thank you for using Synapse. Goodbye!")
                break
            if query.lower() in ("reset", "new", "clear"):
                engine.reset()
                print("  [conversation reset]\n")
                continue
            if not query:
                continue

            answer = engine.ask(query)
            print("\n" + "-" * 80)
            print(answer)
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n  Session interrupted. Goodbye!")
            break
        except Exception as exc:
            print(f"\n  Error: {exc}")
            print("  Please try again.\n")


def main():
    """Entry point: initialise Synapse and start interactive mode."""
    try:
        engine = SynapseEngine()
        run_interactive(engine)
    except Exception as exc:
        print(f"  Failed to initialise Synapse: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
