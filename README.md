# Fault Propagation Graph (FPG) Schema

Representation spec and code definitions for fault propagation graphs in fault-injection
scenarios. The code is the spec: all structures, value spaces, and validation rules
live in `src/fpg` as the single source of truth.

## Architecture: invariant structural layer + variable vocabulary layer

```
Invariant (structural layer, core code)            Variable (vocabulary layer, one profile per system)
───────────────────────────────────────            ───────────────────────────────────────────────────
Three node shapes event/precondition/gate          node predicates (differ per system)
Graph-level rules (DAG/no in-edges/time            edge mechanisms
  consistency…)                                    entity types
Combine/Grounding/VerificationLevel                (all defined by each system's own profile)
Evidence structure (query + explanation)
                  │                                        │
                  └──────── build_schema(profile) ─────────┘
                            (factory)
                                  ↓
                  system-specific Scenario / ModelRCAOutput
                  (predicate/mechanism bound to that system's vocabulary enums)
```

- **Structural layer** (`fpg.scenario` / `fpg.model_output`): vocabulary-agnostic;
  `predicate`/`mechanism` are only checked for snake_case format. All graph-level rules
  live here and are inherited by the generated models;
- **Vocabulary layer** (`fpg.profile.VocabProfile`): one config file per system
  (TOML/JSON/YAML), maintained by each system itself. **The core profile is empty** —
  it ships no predefined vocabulary and serves only as a version-lineage anchor
  (`extends = "core"` attaches a profile to this lineage);
- **Factory** (`fpg.factory.build_schema`): dynamically turns a profile into
  vocabulary-bound pydantic models (`SchemaBundle`); the `other` escape hatch is
  auto-injected into every generated enum.

## Directory layout

```
pyproject.toml                    Package definition (uv, src layout)
src/fpg/                          Core package: structural definitions only (single source of truth for the schema)
  version.py                      SCHEMA_VERSION (structural-layer version) and evolution policy
  vocab.py                        Invariant enums + VocabEntry/Stability registry types
  profile.py                      VocabProfile (carrier of the variable layer) + load_profile()
  profiles/core.toml              Empty core profile (version-lineage anchor, package data)
  factory.py                      build_schema(profile) → SchemaBundle
  entities.py                     Entity-type registry mechanism (data lives in profiles)
  types.py                        Base types: EntityRef / TimeInterval / Evidence(+Query)
  scenario.py                     Structural-layer ground truth schema (all graph-level validation)
  model_output.py                 Structural-layer model output contract (output structure for evaluated models)
config/                           Per-system vocabulary config files (conventional location, one per system)
  template.toml                   Template documenting every available field; copy and rename to use
docs/                             Documentation
```

JSON Schema is not maintained on disk; generate it at runtime when needed:
`schema.Scenario.model_json_schema()` (the generated enums carry the system's full
vocabulary and can be dropped straight into an LLM prompt).

## Vocabulary config files (one per system, self-maintained)

Start by copying [config/template.toml](config/template.toml) — it documents every
available field (including optional ones like `stability`/`since`/`renamed_to`/`parent`).
Minimal example:

```toml
# config/bigdata.toml
name = "bigdata"
version = "0.1.0"
extends = "core"                  # attach to the core version lineage (core itself is empty); duplicate names = error

[node_predicates.job_failed]
brief = "Batch job terminated with failure"

[edge_mechanisms.shuffle_backpressure]
brief = "Shuffle fetch slowness backpressures upstream stages"

[entity_types.stage]
brief = "Job stage"
parent = "job"                    # granularity hierarchy, used by scoring for granularity decay
```

```python
from fpg import build_schema, load_profile

schema = build_schema(load_profile("config/bigdata.toml"))
schema.profile.vocab_version          # "core-0.1.0+bigdata-0.1.0"
scenario = schema.Scenario.model_validate_json(raw)   # vocabulary membership enforced here
schema.vocab_for_model()              # predicate vocabulary handed to the evaluated model
schema.entity_registry                # entity types declared in this system's profile
```

The top-level `fpg.Scenario` / `fpg.ModelRCAOutput` are **structural-layer models**:
they validate everything except vocabulary membership (predicate/mechanism are only
checked for snake_case format). To enforce vocabulary, use
`build_schema(load_profile(...))` as shown above.

## Versioning and evolution (modeled on OpenTelemetry semantic conventions)

Two independent semver lines; see `src/fpg/version.py` for the full policy:

- **`SCHEMA_VERSION`** (currently 0.1.0): version of the structural layer.
  MAJOR = breaking change; MINOR = additive fields/constraints; PATCH = documentation
  clarifications only;
- **Vocabulary versions**: live entirely in profiles — each system profile's `version`
  field evolves independently (core is empty; `CORE_PROFILE.version` is only a lineage
  anchor). Adding an entry = MINOR; removing an entry = MAJOR (and must go through a
  deprecation cycle first).

Vocabulary entry lifecycle (semconv-style): new entries enter as `experimental` →
promoted to `stable` after validation against real annotated data → on rename, the old
value is marked `deprecated` with `renamed_to` filled in, kept for at least one MINOR
release, and only removed in the next MAJOR.

**Scenario files are self-describing**: they must carry `schema_version` (semver) and
`vocab_version` (i.e. `profile.vocab_version`, e.g. `core-0.1.0+bigdata-0.1.0`);
consumers use these to select a profile and perform migrations.

## Usage

```bash
uv sync                           # install the package and dev dependencies
```

```python
from fpg import Scenario, ModelRCAOutput

# Parsing is validation: if pydantic validation passes, the file is valid (all graph-level rules built in)
scenario = Scenario.model_validate_json(open("scenario.json").read())
scenario.graph.root_causes        # root causes = nodes with no in-edges (injection points + preconditions), excluding isolated distractors

answer = ModelRCAOutput.model_validate_json(raw_model_answer)
```

## Entity type extension

Entity **types** (prefixes) form an open set, and **all the data lives in profiles**
(fully isomorphic to the vocabulary): each system declares its entity types in its own
profile's `[entity_types.*]` sections (e.g. `svc`/`pod` or `job`/`stage`; `parent`
declares the granularity hierarchy used by scoring decay), which the bundle materializes
into a registry (`schema.entity_registry`). `fpg.entities` provides only the
mechanism (registry + entry point discovery) and carries no data; the entry point
(`[project.entry-points."fpg.entity_types"]`) remains as a supplementary
"install-to-register" channel.

Validation is two-tiered: `EntityRef`'s structural check only verifies the
`<prefix>:<name>` format (schema validation does not depend on which extensions are
installed); whether a prefix is registered is a strict-tier check, performed by
downstream tooling via `schema.entity_registry.unregistered_prefixes(refs)`. Entity
**instances** (which concrete services exist) are testbed data and stay out of the schema.

## Evidence re-execution

Evidence is not a passive locator but a **replayable query**:
`Evidence = {query, explanation}`, where `query = {language, statement}`. The
verification protocol is: re-execute `statement` according to `language`, then check
the result against `explanation`. For example:

```json
{
  "explanation": "pool.max == 10, below capacity-plan requirement of 50",
  "query": {
    "language": "sql",
    "statement": "SELECT value FROM config_snapshots WHERE config_key = 'order-service.datasource.pool.max' AND ts <= '2026-06-01T09:00:00Z' ORDER BY ts DESC LIMIT 1"
  }
}
```

## Value space quick reference

| Field | Value space | Defined in |
|---|---|---|
| `node.predicate` | **Profile-dependent**: entries declared by the system profile + auto-injected `other`; structural layer only enforces snake_case | Data: `config/<system>.toml`; binding: `factory.build_schema` |
| `edge.mechanism` | **Profile-dependent**: entries declared by the system profile + `other`; structural layer only enforces snake_case | Data: `config/<system>.toml`; binding: `factory.build_schema` |
| `evidence.explanation` | Free text (what the query result proves) | `types.Evidence` |
| `evidence.query.language` | `sql \| promql \| logql \| http \| other` (invariant enum; adding values as needed = structural-layer MINOR) | `vocab.QueryLanguage` |
| `evidence.query.statement` | Free-form string (executable query text, e.g. PromQL/SQL/HTTP request line) | `types.EvidenceQuery` |
| `node.kind` | `event \| precondition \| gate` (discriminator of the tagged union; event = something that happened within the fault window (good or bad, instantaneous or sustained), precondition = a latent hazard/premise that existed before the fault, gate = logic gate) (invariant) | `scenario.EventNode / PreconditionNode / GateNode` |
| `node.grounding` | `observed \| latent` (event/precondition only; gates have no such field) (invariant) | `vocab.Grounding` |
| `node.combine` | `AND \| OR` (required when in-edges ≥ 2) (invariant) | `vocab.Combine` |
| `node.annotation` | `auto \| human \| replay-verified` (invariant) | `vocab.AnnotationSource` |
| `edge.verification` | `interventional \| consistency-checked` (invariant) | `vocab.VerificationLevel` |
| `*.subject` / `target_entity` | `<prefix>:<name>`, where prefix = an entity type declared by the system profile (open set) | `types.EntityRef` + `entities` |
| `*.time` | `[start, end]` ISO 8601 interval; instantaneous events have start == end | `types.TimeInterval` |
| `scenario.schema_version` | semver (`^\d+\.\d+\.\d+$`) | `scenario.Scenario` |
| `scenario.vocab_version` | `core-<semver>[+<system>-<semver>]` = `profile.vocab_version` | `profile.VocabProfile` |
| `injection.fault_type` | Free-form string (value space = the injection tool's fault catalog; no enum enforced) | `scenario.Injection` |

Structural constraints beyond the enums (DAG, preconditions have no in-edges, time
consistency, observed nodes require evidence, isolated distractors have degree 0, etc.)
all live in the structural layer (`fpg.scenario` / `fpg.model_output`) and are inherited
by the factory-generated models; see the `scenario.py` module docstring for the full list.
