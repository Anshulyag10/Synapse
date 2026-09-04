# Knowledge Graph

## Schema

```
(:Disease {disease_code, name, contagious, chronic, raw_treatments})
(:Symptom {name, aliases[]})
(:Treatment {name, aliases[]})

(Disease)-[:HAS_SYMPTOM]->(Symptom)
(Disease)-[:TREATED_BY]->(Treatment)
```

## Ingestion Pipeline

```
disease_data.csv
    │
    ▼
OllamaNormalizer (per-record LLM normalisation)
    │  ◄── few-shot prompting with qwen2.5:7b-instruct
    │  ◄── disk cache (.llm_cache.json) for repeat runs
    │
    ▼
EntityResolver (global fuzzy deduplication)
    │  ◄── symptom threshold: 90
    │  ◄── treatment threshold: 90
    │
    ▼
Neo4j Writer
    │  ◄── MERGE with uniqueness constraints
    │  ◄── batch upserts (500 per batch)
    │
    ▼
Knowledge Graph (Disease → Symptom, Disease → Treatment)
```

### Entity Normalisation

Each disease's free-text symptoms and treatments are atomised by a local
Ollama model:
- "High fever, dry cough, body aches" → `fever`, `cough`, `body ache`
- Aliases are generated: `fever` → `["high fever", "pyrexia"]`
- Canonical names are ≤4 words, lowercase

### Entity Resolution

After per-record normalisation, a global pass merges synonymous entities:
- Greedy fuzzy clustering using `token_sort_ratio`
- Threshold 90 for both symptoms and treatments
- Result: one canonical node per concept, aliases attached

### Graph Statistics (healthcare domain)

After full ingestion of the included `disease_data.csv`:
- ~400 Disease nodes
- ~600 Symptom nodes
- ~400 Treatment nodes
- ~2,000 HAS_SYMPTOM relationships
- ~800 TREATED_BY relationships

## Retrieval

See [retrieval.md](retrieval.md) for the full retrieval pipeline.
