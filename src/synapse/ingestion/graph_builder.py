"""Knowledge-graph builder: normalise CSV records and write to Neo4j.

Pipeline:
  CSV → LLM normalisation → entity resolution → Neo4j upserts

Node types created:
  (:Disease {disease_code, name, contagious, chronic, raw_treatments})
  (:Symptom {name, aliases})
  (:Treatment {name, aliases})

Relationships:
  (Disease)-[:HAS_SYMPTOM]->(Symptom)
  (Disease)-[:TREATED_BY]->(Treatment)
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase

from synapse.core.config import SynapseSettings, load_settings
from synapse.core.logging import get_logger
from synapse.ingestion.normalization import (
    EntityResolver,
    OllamaNormalizer,
    clean_entity_list,
    naive_fallback,
)

logger = get_logger("ingestion.graph")


@dataclass
class DiseaseRecord:
    """Intermediate representation of a normalised disease."""
    disease_code: str
    name: str
    contagious: bool
    chronic: bool
    raw_treatments: str
    symptoms: list[dict] = field(default_factory=list)
    treatments: list[dict] = field(default_factory=list)


class Neo4jWriter:
    """Thin Neo4j writer with reconnect/retry for graph upserts."""

    def __init__(self, settings: SynapseSettings):
        self._cfg = settings
        self.driver = None

    def connect(self):
        import time
        for attempt in range(3):
            try:
                if self.driver:
                    try:
                        self.driver.close()
                    except Exception:
                        pass
                self.driver = GraphDatabase.driver(
                    self._cfg.neo4j_uri,
                    auth=(self._cfg.neo4j_username, self._cfg.neo4j_password),
                    keep_alive=True,
                    max_connection_lifetime=300,
                    connection_acquisition_timeout=30,
                )
                with self.driver.session(database=self._cfg.neo4j_database) as s:
                    s.run("RETURN 1")
                logger.info("Neo4j connection established")
                return
            except Exception as exc:
                logger.warning("Connect attempt %d/3 failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError("Could not connect to Neo4j after 3 attempts")

    def _run(self, cypher: str, retries: int = 3, **params):
        import time
        for attempt in range(retries):
            try:
                with self.driver.session(database=self._cfg.neo4j_database) as s:
                    s.run(cypher, **params)
                return
            except Exception as exc:
                if attempt < retries - 1:
                    logger.warning("Write error (attempt %d): %s", attempt + 1, exc)
                    time.sleep(3)
                    self.connect()
                else:
                    raise

    def setup(self, clear: bool):
        if clear:
            self._run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared")
        self._run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.disease_code IS UNIQUE")
        self._run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE")
        self._run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE")
        logger.info("Constraints ready")

    def upsert_symptom_nodes(self, symptoms: list[dict]):
        for i in range(0, len(symptoms), 500):
            batch = symptoms[i : i + 500]
            self._run(
                """
                UNWIND $rows AS row
                MERGE (n:Symptom {name: row.name})
                SET n.aliases = row.aliases
                """,
                rows=batch,
            )

    def upsert_treatment_nodes(self, treatments: list[dict]):
        for i in range(0, len(treatments), 500):
            batch = treatments[i : i + 500]
            self._run(
                """
                UNWIND $rows AS row
                MERGE (n:Treatment {name: row.name})
                SET n.aliases = row.aliases
                """,
                rows=batch,
            )

    def upsert_disease(
        self, disease: DiseaseRecord, symptom_names: list[str], treatment_names: list[str]
    ):
        self._run(
            """
            MERGE (d:Disease {disease_code: $code})
            SET d.name = $name,
                d.contagious = $contagious,
                d.chronic = $chronic,
                d.raw_treatments = $raw_treatments
            WITH d
            UNWIND $symptoms AS sname
            MATCH (sym:Symptom {name: sname})
            MERGE (d)-[:HAS_SYMPTOM]->(sym)
            """,
            code=disease.disease_code,
            name=disease.name,
            contagious=disease.contagious,
            chronic=disease.chronic,
            raw_treatments=disease.raw_treatments,
            symptoms=symptom_names,
        )
        if treatment_names:
            self._run(
                """
                MATCH (d:Disease {disease_code: $code})
                UNWIND $treatments AS tname
                MATCH (t:Treatment {name: tname})
                MERGE (d)-[:TREATED_BY]->(t)
                """,
                code=disease.disease_code,
                treatments=treatment_names,
            )

    def stats(self) -> dict:
        with self.driver.session(database=self._cfg.neo4j_database) as s:
            r = s.run(
                """
                OPTIONAL MATCH (d:Disease) WITH count(DISTINCT d) AS diseases
                OPTIONAL MATCH (sy:Symptom) WITH diseases, count(DISTINCT sy) AS symptoms
                OPTIONAL MATCH (t:Treatment) WITH diseases, symptoms, count(DISTINCT t) AS treatments
                OPTIONAL MATCH ()-[h:HAS_SYMPTOM]->() WITH diseases, symptoms, treatments, count(h) AS has_symptom
                OPTIONAL MATCH ()-[tb:TREATED_BY]->()
                RETURN diseases, symptoms, treatments, has_symptom, count(tb) AS treated_by
                """
            ).single()
            return dict(r)

    def close(self):
        if self.driver:
            self.driver.close()


class GraphBuildPipeline:
    """Full pipeline: normalise CSV → resolve entities → write graph."""

    def __init__(self, settings: Optional[SynapseSettings] = None):
        self._cfg = settings or load_settings()
        self._normalizer = OllamaNormalizer(self._cfg)
        self._symptom_resolver = EntityResolver(self._cfg.symptom_merge_threshold)
        self._treatment_resolver = EntityResolver(self._cfg.treatment_merge_threshold)
        self._writer = Neo4jWriter(self._cfg)

    def run(self, csv_path: str, clear_existing: bool = True):
        logger.info("Starting knowledge-graph ingestion")
        self._check_ollama()
        rows = self._load_csv(csv_path)

        # Phase 1: normalise
        diseases: list[DiseaseRecord] = []
        for i, row in enumerate(rows, 1):
            name = row.get("Name", "")
            symptoms_raw = row.get("Symptoms", "")
            treatments_raw = row.get("Treatments", "")

            structured = self._normalizer.normalize(name, symptoms_raw, treatments_raw)
            if not structured:
                logger.warning("Fallback normalisation for: %s", name)
                structured = naive_fallback(name, symptoms_raw, treatments_raw)

            diseases.append(
                DiseaseRecord(
                    disease_code=(row.get("Disease_Code", "") or "").strip().lower(),
                    name=(structured.get("record_name") or structured.get("disease_name") or name).strip().lower(),
                    contagious=str(row.get("Contagious", "False")).strip().lower() == "true",
                    chronic=str(row.get("Chronic", "False")).strip().lower() == "true",
                    raw_treatments=treatments_raw.strip(),
                    symptoms=clean_entity_list(structured.get("symptoms")),
                    treatments=clean_entity_list(structured.get("treatments")),
                )
            )
            if i % 25 == 0:
                logger.info("Normalised %d/%d", i, len(rows))

        # Phase 2: resolve entities
        logger.info("Resolving entities (global dedup)")
        disease_symptoms: dict[str, list[str]] = {}
        disease_treatments: dict[str, list[str]] = {}
        for d in diseases:
            s_names = []
            for ent in d.symptoms:
                canon = self._symptom_resolver.resolve(ent["name"], ent["aliases"])
                if canon and canon not in s_names:
                    s_names.append(canon)
            disease_symptoms[d.disease_code] = s_names

            t_names = []
            for ent in d.treatments:
                canon = self._treatment_resolver.resolve(ent["name"], ent["aliases"])
                if canon and canon not in t_names:
                    t_names.append(canon)
            disease_treatments[d.disease_code] = t_names

        symptom_nodes = [
            {"name": n, "aliases": self._symptom_resolver.aliases_for(n)}
            for n in self._symptom_resolver.canonical
        ]
        treatment_nodes = [
            {"name": n, "aliases": self._treatment_resolver.aliases_for(n)}
            for n in self._treatment_resolver.canonical
        ]
        logger.info(
            "Canonical: %d symptoms, %d treatments",
            len(symptom_nodes),
            len(treatment_nodes),
        )

        # Phase 3: write graph
        self._writer.connect()
        self._writer.setup(clear=clear_existing)
        self._writer.upsert_symptom_nodes(symptom_nodes)
        self._writer.upsert_treatment_nodes(treatment_nodes)
        for d in diseases:
            self._writer.upsert_disease(
                d,
                disease_symptoms.get(d.disease_code, []),
                disease_treatments.get(d.disease_code, []),
            )

        stats = self._writer.stats()
        logger.info("Knowledge graph complete")
        for k, v in stats.items():
            logger.info("  %s: %s", k, v)

    def _load_csv(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        logger.info("Loaded %d rows from %s", len(rows), path)
        return rows

    def _check_ollama(self):
        try:
            import ollama
            available = [m.get("model", "") for m in ollama.list().get("models", [])]
            if not any(self._normalizer.model.split(":")[0] in a for a in available):
                logger.warning(
                    "Model '%s' not found in Ollama. Run: ollama pull %s",
                    self._normalizer.model,
                    self._normalizer.model,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama ({exc}). Install from https://ollama.com"
            )

    def close(self):
        self._writer.close()


def main():
    """Entry point: ingest disease data into the knowledge graph."""
    cfg = load_settings()
    csv_path = Path(cfg.data_dir) / "disease_data.csv"
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return
    pipeline = GraphBuildPipeline(cfg)
    try:
        pipeline.run(str(csv_path), clear_existing=True)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
