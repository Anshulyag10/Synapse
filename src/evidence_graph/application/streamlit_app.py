"""Streamlit frontend for EvidenceGraph.

A clean, evidence-transparent chat interface.  The sidebar shows the
retrieval strategy and evidence sources.  Each answer displays its
grounding confidence and the retrieval path taken.
"""

import os
import io
import re
import sys
import logging
import warnings
import contextlib
from pathlib import Path

import streamlit as st

# Silence third-party noise
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT.parent / ".env")

from evidence_graph.application.cli import EvidenceGraphEngine


# ── Theming ────────────────────────────────────────────────────────────

PALETTE = {
    "primary": "#5B5FC7",
    "deep": "#3A3E8C",
    "light": "#EDEEF9",
    "ink": "#1A1D3D",
    "border": "#D0D2E8",
    "surface": "#FFFFFF",
    "accent": "#7B68EE",
}

CSS = f"""
<style>
.block-container {{ max-width: 860px; padding-top: 2.2rem; }}
#eg-head {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.1rem; }}
#eg-head .dot {{
    width:12px; height:12px; border-radius:50%;
    background:{PALETTE['primary']}; box-shadow:0 0 0 4px {PALETTE['light']};
}}
#eg-head h1 {{ font-size:1.7rem; margin:0; color:{PALETTE['deep']}; letter-spacing:.3px; }}
#eg-sub {{ color:#6B6F9C; margin:.1rem 0 1.1rem; font-size:.95rem; }}
[data-testid="stChatMessage"] {{
    background:{PALETTE['surface']}; border:1px solid {PALETTE['border']};
    border-radius:14px; padding:.4rem .9rem;
}}
.stChatMessage p {{ color:{PALETTE['ink']}; }}
[data-testid="stStatusWidget"] {{ border-radius:12px; }}
hr {{ border-color:{PALETTE['border']}; }}
.disclaimer {{ color:#8B8FB0; font-size:.82rem; }}
.evidence-badge {{
    display:inline-block; padding:2px 8px; border-radius:6px;
    font-size:.75rem; font-weight:600; margin-right:4px;
}}
.badge-graph {{ background:#D4F0DE; color:#1D6E3A; }}
.badge-vector {{ background:#D4E0F0; color:#1D4D6E; }}
.badge-lexical {{ background:#F0E4D4; color:#6E4D1D; }}
</style>
"""

# ── Progress prettifier ────────────────────────────────────────────────

TOOL_LABELS = {
    "retrieve_entities_by_attributes": "Searching the knowledge graph",
    "get_entity_details": "Reading the entity record",
    "get_document_details": "Looking up document details",
    "search_corpus": "Searching the document corpus",
}

RULES = [
    (r"\[tool\*?\]\s*(?:recovered\s*)?([\w]+)", lambda m: TOOL_LABELS.get(m.group(1))),
    (r"Initialising EvidenceGraph", "Starting EvidenceGraph"),
    (r"Connected to Neo4j", "Connected to the knowledge graph"),
    (r"Connected to ChromaDB", "Connected to the document index"),
    (r"Loading embedding", "Loading the embedding model"),
    (r"\[Reranker\] Loading", "Loading the reranking model"),
    (r"Evidence agent ready", "Reasoning agent ready"),
    (r"^Ready!?$", "Ready"),
    (r"Attribute retrieval", "Matching attributes in the knowledge graph"),
    (r"Semantic search", "Searching the document index"),
    (r"Expanded to (\d+)", lambda m: f"Found {m.group(1)} related entities"),
    (r"Retrieved (\d+)", lambda m: f"Retrieved {m.group(1)} documents"),
    (r"Query classified as (\w+)", lambda m: f"Query category: {m.group(1)}"),
]

NOISE = ("you:", "thank you", "interactive mode", "seeds:")


def prettify(line: str):
    line = line.strip()
    if not line or not any(c.isalnum() for c in line):
        return None
    if any(n in line.lower() for n in NOISE):
        return None
    for pattern, repl in RULES:
        m = re.search(pattern, line)
        if m:
            return repl(m) if callable(repl) else repl
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", line)


class UIStream(io.TextIOBase):
    SKIP = ("warn", "error", "fail", "traceback", "exception")

    def __init__(self, on_step):
        self.on_step = on_step
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, line):
        if any(k in line.lower() for k in self.SKIP):
            return
        step = prettify(line)
        if step:
            self.on_step(step)


@contextlib.contextmanager
def captured(on_step):
    with contextlib.redirect_stdout(UIStream(on_step)), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ── Engine ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine(_on_step):
    with captured(_on_step):
        engine = EvidenceGraphEngine()
    return engine


# ── Page ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EvidenceGraph",
    page_icon="🔍",
    layout="centered",
)
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div id="eg-head"><span class="dot"></span><h1>EvidenceGraph</h1></div>'
    '<div id="eg-sub">Evidence-grounded answers from knowledge graphs and documents.</div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Session")
    if st.button("New conversation", use_container_width=True):
        if "engine" in st.session_state:
            st.session_state.engine.reset()
        st.session_state.history = []
        st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Retrieval Sources")
    st.markdown(
        '<span class="evidence-badge badge-graph">Knowledge Graph</span>'
        '<span class="evidence-badge badge-vector">Semantic Index</span>'
        '<span class="evidence-badge badge-lexical">Lexical Index</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p class="disclaimer">Educational use only — not professional advice. '
        "Consult a qualified professional.</p>",
        unsafe_allow_html=True,
    )

# Boot
if "engine" not in st.session_state:
    with st.status("Starting EvidenceGraph…", expanded=True) as boot:
        try:
            st.session_state.engine = get_engine(lambda step: st.write(f"• {step}"))
            boot.update(label="EvidenceGraph is ready", state="complete", expanded=False)
        except Exception:
            boot.update(label="Could not start EvidenceGraph", state="error")
            st.error(
                "EvidenceGraph could not connect to its data stores. "
                "Check Neo4j and ChromaDB setup, then reload the page."
            )
            st.stop()

engine = st.session_state.engine

# Replay history
for role, text in st.session_state.history:
    with st.chat_message(role, avatar="🔍" if role == "assistant" else None):
        st.markdown(text)

# Input
prompt = st.chat_input("Ask about symptoms, diseases, medicines, or drug interactions…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔍"):
        answer_box = st.empty()
        with st.status("Thinking…", expanded=True) as status:
            try:
                with captured(lambda step: st.write(f"• {step}")):
                    answer = engine.ask(prompt)
                status.update(label="Answer ready", state="complete", expanded=False)
            except Exception:
                status.update(label="Something went wrong", state="error")
                answer = "Sorry, I ran into a problem answering that. Please try again."
        answer_box.markdown(answer)

    st.session_state.history.append(("assistant", answer))
