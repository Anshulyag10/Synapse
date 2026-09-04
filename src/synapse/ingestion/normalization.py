"""Entity normalisation: LLM-based atomisation and synonym resolution.

Extracts atomic, canonical entities from free-text fields using a local
LLM (Ollama).  Includes:
  - OllamaNormalizer: per-record LLM normalisation with caching
  - EntityResolver: greedy fuzzy clustering across all records
  - naive_fallback: comma-split fallback when LLM normalisation fails
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from synapse.core.config import SynapseSettings, load_settings
from synapse.core.logging import get_logger

logger = get_logger("ingestion.normalization")


class OllamaNormalizer:
    """Normalise a record's free-text attributes into structured entities.

    Uses a local Ollama model with few-shot examples to atomise compound
    phrases, canonicalise names, and generate aliases.  Results are cached
    to disk so repeated runs skip the LLM.
    """

    SYSTEM_PROMPT = """You are a knowledge-graph normalization assistant.
You convert ONE record's free-text attributes into clean structured JSON
for insertion into a graph database.

THE GRAPH IS MANY-TO-MANY:
- One free-text field becomes MANY atomic nodes (split it apart).
- The SAME concept can belong to MANY records. Give a shared concept the
  SAME canonical "name" every time so it collapses into ONE shared node.

RULES:

1. ATOMIZE — split every compound field into individual items.
   "nausea, vomiting, abdominal pain" → three nodes.

2. SPLIT "X or Y" AND "X and Y" COMPOUNDS into two separate nodes each.
   "nausea or vomiting" → two nodes: "nausea" and "vomiting".

3. EXPAND "X (such as A, B, C)" by creating A, B, C as INDIVIDUAL nodes.
   Drop the generic parent X if A/B/C are specific enough.

4. CANONICALIZE the name to the shortest standard term (MAX 4 WORDS).
   Strip severity/frequency/location qualifiers to aliases.
   "High fever" → name "fever", alias "high fever".

5. ALIASES = well-known synonyms + original surface wordings you stripped.
   Lowercase. 2-4 aliases max. Empty list [] if none.

6. Lowercase all names and aliases. Trim whitespace. Remove duplicates.

7. Empty fields → return [].

8. Output ONLY the JSON object. No commentary, no markdown, no code fences.

JSON schema:
{
  "record_name": "normalized name, lowercase",
  "symptoms":   [{"name": "canonical", "aliases": ["synonym1"]}],
  "treatments": [{"name": "canonical", "aliases": ["synonym1"]}]
}"""

    FEW_SHOT = [
        {
            "role": "user",
            "content": (
                "Record: Influenza\n"
                "Symptoms: High fever, dry cough, body aches and severe fatigue\n"
                "Treatments: Rest, plenty of fluids, antiviral medication (in severe cases)\n\n"
                "Return ONLY the JSON object."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "record_name": "influenza",
                    "symptoms": [
                        {"name": "fever", "aliases": ["high fever", "pyrexia"]},
                        {"name": "cough", "aliases": ["dry cough"]},
                        {"name": "body ache", "aliases": ["myalgia", "body aches"]},
                        {"name": "fatigue", "aliases": ["tiredness", "severe fatigue"]},
                    ],
                    "treatments": [
                        {"name": "rest", "aliases": []},
                        {"name": "fluids", "aliases": ["hydration", "oral rehydration"]},
                        {"name": "antiviral medication", "aliases": ["antivirals", "oseltamivir"]},
                    ],
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": (
                "Record: Malaria\n"
                "Symptoms: Recurrent fever with chills, sweating, headache and fatigue\n"
                "Treatments: Antimalarial drugs such as chloroquine or artemisinin, supportive care\n\n"
                "Return ONLY the JSON object."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "record_name": "malaria",
                    "symptoms": [
                        {"name": "fever", "aliases": ["pyrexia", "recurrent fever"]},
                        {"name": "chills", "aliases": ["shivering", "rigors"]},
                        {"name": "sweating", "aliases": ["diaphoresis"]},
                        {"name": "headache", "aliases": ["cephalgia"]},
                        {"name": "fatigue", "aliases": ["tiredness"]},
                    ],
                    "treatments": [
                        {"name": "chloroquine", "aliases": ["chloroquine phosphate"]},
                        {"name": "artemisinin", "aliases": ["artemisinin-based therapy"]},
                        {"name": "supportive care", "aliases": []},
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    def __init__(self, settings: Optional[SynapseSettings] = None):
        cfg = settings or load_settings()
        import ollama

        self.client = ollama
        self.model = cfg.ollama_model
        self._cache: dict[str, dict] = {}
        self._cache_path = Path(cfg.data_dir).parent / ".llm_cache.json"
        self._load_cache()

    def _load_cache(self):
        if self._cache_path.exists():
            try:
                self._cache = json.loads(
                    self._cache_path.read_text(encoding="utf-8")
                )
                logger.info("Loaded %d cached normalizations", len(self._cache))
            except Exception as exc:
                logger.warning("Cache load failed: %s", exc)
                self._cache = {}

    def _save_cache(self):
        try:
            self._cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Cache save failed: %s", exc)

    def _cache_key(self, name: str, symptoms: str, treatments: str) -> str:
        return hashlib.md5(
            f"{self.model}|{name}|{symptoms}|{treatments}".encode()
        ).hexdigest()

    def normalize(
        self, name: str, symptoms_raw: str, treatments_raw: str
    ) -> Optional[dict]:
        """Normalize one record's attributes to structured JSON (cached)."""
        key = self._cache_key(name, symptoms_raw, treatments_raw)
        if key in self._cache:
            return self._cache[key]

        user_prompt = (
            f"Record: {name}\n"
            f"Symptoms: {symptoms_raw}\n"
            f"Treatments: {treatments_raw}\n\n"
            "Return ONLY the JSON object."
        )
        messages = (
            [{"role": "system", "content": self.SYSTEM_PROMPT}]
            + self.FEW_SHOT
            + [{"role": "user", "content": user_prompt}]
        )
        try:
            resp = self.client.chat(
                model=self.model,
                messages=messages,
                format="json",
                options={"temperature": 0},
            )
            content = resp["message"]["content"].strip()
            result = json.loads(content)
            self._cache[key] = result
            self._save_cache()
            return result
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error for '%s': %s", name, exc)
            return None
        except Exception as exc:
            logger.error("Ollama error for '%s': %s", name, exc)
            raise


def clean_entity_list(raw: list) -> list[dict]:
    """Coerce LLM output into [{name, aliases[]}], lowercased and deduped."""
    out, seen = [], set()
    for item in raw or []:
        if isinstance(item, str):
            name, aliases = item, []
        elif isinstance(item, dict):
            name, aliases = item.get("name", ""), item.get("aliases", []) or []
        else:
            continue
        name = str(name).strip().lower().replace("_", " ")
        if not name or name in seen:
            continue
        seen.add(name)
        clean_aliases = sorted(
            {str(a).strip().lower().replace("_", " ") for a in aliases if str(a).strip()}
        )
        out.append({"name": name, "aliases": clean_aliases})
    return out


def naive_fallback(name: str, symptoms_raw: str, treatments_raw: str) -> dict:
    """Comma-split fallback when LLM normalisation fails."""
    def split_func(s: str) -> list:
        return [
            {"name": p.strip().lower(), "aliases": []}
            for p in s.split(",")
            if p.strip()
        ]
    return {
        "record_name": name.strip().lower(),
        "symptoms": split_func(symptoms_raw),
        "treatments": split_func(treatments_raw),
    }


class EntityResolver:
    """Greedy fuzzy clustering of names into canonical entities with aliases."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.canonical: dict[str, set] = {}
        self.lookup: dict[str, str] = {}

    def resolve(self, name: str, aliases: list[str]) -> str:
        """Register an entity (merging into a fuzzy match) and return its canonical name."""
        name = name.strip().lower().replace("_", " ")
        if not name:
            return ""

        if name in self.lookup:
            canon = self.lookup[name]
        else:
            canon = self._find_match(name)
            if canon is None:
                canon = name
                self.canonical[canon] = set()
            self.canonical[canon].add(name)
            self.lookup[name] = canon

        for a in aliases:
            a = a.strip().lower().replace("_", " ")
            if a and a != canon:
                self.canonical[canon].add(a)
                self.lookup.setdefault(a, canon)
        return canon

    def _find_match(self, name: str) -> Optional[str]:
        best, best_score = None, 0
        for canon in self.canonical:
            score = fuzz.token_sort_ratio(name, canon)
            if score > best_score:
                best, best_score = canon, score
        return best if best_score >= self.threshold else None

    def aliases_for(self, canonical_name: str) -> list[str]:
        return sorted(self.canonical.get(canonical_name, set()) - {canonical_name})
