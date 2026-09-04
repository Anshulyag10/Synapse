"""Convenience entry point for Synapse.

Usage:
  python run.py                   # interactive QA REPL
  python run.py serve             # start FastAPI server
  python run.py ingest-graph      # run knowledge graph ingestion
  python run.py ingest-vector     # run vector store ingestion
"""

import sys
from pathlib import Path

# Ensure the src package is importable
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main():
    """Route to the appropriate subcommand."""
    command = sys.argv[1] if len(sys.argv) > 1 else "query"

    if command == "query":
        from synapse.application.cli import main as cli_main
        cli_main()

    elif command == "serve":
        import uvicorn
        from synapse.core.config import load_settings

        cfg = load_settings()
        uvicorn.run(
            "synapse.application.api:app",
            host=cfg.api_host,
            port=cfg.api_port,
            reload=False,
        )

    elif command == "ingest-graph":
        from synapse.ingestion.graph_builder import main as graph_main
        graph_main()

    elif command == "ingest-vector":
        from synapse.ingestion.vector_indexer import main as vector_main
        vector_main()

    elif command == "benchmark":
        from synapse.evaluation.benchmark import main as bench_main
        bench_main()

    elif command == "streamlit":
        import subprocess
        app_path = Path(__file__).parent / "src" / "synapse" / "application" / "streamlit_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

    else:
        print(f"Unknown command: {command}")
        print("Available commands: query, serve, ingest-graph, ingest-vector, benchmark, streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
