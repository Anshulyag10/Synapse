"""Healthcare domain adapter.

Maps the generic EvidenceGraph retrieval engine to healthcare-specific
concepts: Disease, Symptom, Treatment, Medicine.

The adapter provides:
  - Entity type registry (what kinds of nodes exist in the KG)
  - Relationship type registry
  - Domain-specific prompt context
  - Display-friendly tool descriptions

Swapping this adapter (e.g. for finance or legal) would let the same
core engine operate over a different domain without rewriting retrieval.
"""

from dataclasses import dataclass, field


@dataclass
class EntityTypeConfig:
    """Configuration for one type of entity in the domain."""
    label: str
    description: str
    properties: list[str] = field(default_factory=list)
    searchable: bool = True


@dataclass
class RelationshipConfig:
    """Configuration for one type of relationship in the domain."""
    type_name: str
    from_label: str
    to_label: str
    description: str


# ── Healthcare entity types ────────────────────────────────────────────

ENTITY_TYPES = {
    "Disease": EntityTypeConfig(
        label="Disease",
        description="A medical condition with known symptoms and treatments.",
        properties=[
            "name", "disease_code", "contagious", "chronic", "raw_treatments",
        ],
    ),
    "Symptom": EntityTypeConfig(
        label="Symptom",
        description="An observable sign or reported experience indicating a condition.",
        properties=["name", "aliases"],
    ),
    "Treatment": EntityTypeConfig(
        label="Treatment",
        description="A therapeutic intervention for a condition.",
        properties=["name", "aliases"],
    ),
    "Medicine": EntityTypeConfig(
        label="Medicine",
        description="A pharmaceutical product with composition and side effects.",
        properties=[
            "product_name", "salt_composition", "sub_category",
            "manufacturer", "price", "medicine_desc",
            "side_effects", "drug_interactions",
        ],
        searchable=True,
    ),
}

# ── Healthcare relationship types ──────────────────────────────────────

RELATIONSHIP_TYPES = [
    RelationshipConfig(
        type_name="HAS_SYMPTOM",
        from_label="Disease",
        to_label="Symptom",
        description="A disease presents with this symptom.",
    ),
    RelationshipConfig(
        type_name="TREATED_BY",
        from_label="Disease",
        to_label="Treatment",
        description="A disease is treated by this intervention.",
    ),
    RelationshipConfig(
        type_name="USED_FOR",
        from_label="Medicine",
        to_label="Disease",
        description="A medicine is used for treating this disease.",
    ),
    RelationshipConfig(
        type_name="HAS_INTERACTION_WITH",
        from_label="Medicine",
        to_label="Medicine",
        description="Two medicines have a known drug interaction.",
    ),
]

# ── Display-friendly tool labels ───────────────────────────────────────

TOOL_DISPLAY_LABELS = {
    "retrieve_entities_by_attributes": "Searching diseases that match the symptoms",
    "get_entity_details": "Reading the disease record",
    "get_document_details": "Looking up the medicine details",
    "search_corpus": "Searching the medicine database",
}

# ── Domain-specific prompt context ─────────────────────────────────────

DOMAIN_CONTEXT = """Healthcare Domain Context:
The knowledge graph stores diseases linked to symptoms and treatments.
The document corpus contains ~195,000 medicine records with compositions,
descriptions, side effects, and drug interactions.

Data limitations:
- No disease causes, prevention, prognosis, or diagnostic tests.
- No medicine dosage, pregnancy/age-specific safety, or efficacy ranking.
- Data sourced from public datasets — not clinically validated.

Disclaimer: This system is for educational and research purposes only.
It is NOT a clinical diagnostic tool and should not replace professional
medical advice."""

DISCLAIMER = (
    "This is for educational purposes only — not professional medical advice. "
    "Consult a healthcare provider."
)


def get_kg_schema_description() -> str:
    """Return a human-readable description of the healthcare KG schema."""
    lines = ["Knowledge Graph Schema (Healthcare Domain):", ""]
    lines.append("Entity Types:")
    for etype in ENTITY_TYPES.values():
        lines.append(f"  ({etype.label}) — {etype.description}")
        lines.append(f"    Properties: {', '.join(etype.properties)}")
    lines.append("")
    lines.append("Relationships:")
    for rel in RELATIONSHIP_TYPES:
        lines.append(
            f"  (:{rel.from_label})-[:{rel.type_name}]->(:{rel.to_label}) — {rel.description}"
        )
    return "\n".join(lines)
