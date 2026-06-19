# Fault Propagation Graph (FPG) Schema

Schema for representing how an injected fault propagates through a system. A fault
injection experiment yields a ground-truth propagation graph: the injection point is
the root cause, and the disturbance cascades along causal edges until it surfaces as
SLO violations. These graphs form a dataset with verifiable ground truth, against
which LLMs are evaluated on root cause analysis: given raw observation data, the model
must reconstruct the propagation graph and nominate the root causes.

The code is the spec: all structures, value spaces, and validation rules live in
`src/fpg` as the single source of truth.

## Design

### Why a graph, and what a node is

A fault rarely propagates as a chain: one cause fans out to several effects, and one
effect may require several causes jointly (a trigger AND a standing weakness). So the
ground truth is a graph whose nodes are **time-anchored, verifiable statements** —
"entity `subject` exhibited failure mode `predicate` during `[start, end]`" — never
free-form prose. Time anchoring is also what makes the graph a **DAG by construction**:
a cause cannot start after its effect (enforced as a validation rule), and feedback
loops are unrolled over time — the same `(subject, predicate)` recurring later becomes
a new node, never a back-edge. **Root causes are exactly the source nodes** (no
incoming edges): injection points plus preconditions.

### Three node kinds

The discriminated union on `kind` exists because the three kinds have genuinely
different shapes, not just different labels:

- **`event`** — something that happened on a real entity during the fault window
  (good or bad, instantaneous or sustained). Fully grounded: subject, predicate,
  time, evidence.
- **`precondition`** — a standing weakness that existed *before* the fault and
  enabled it (undersized pool, missing timeout). The axis that separates it from
  `event` is **causal origin, not duration**: it was already there at injection time,
  so it can never be caused by this fault — structurally, it has no incoming edges.
  The typical pattern is "trigger event AND precondition → effect".
- **`gate`** — a pure boolean connector for mixed expressions like `(A AND B) OR C`,
  which a single per-node `combine` (AND/OR over all in-edges) cannot express. It is
  not a statement about the system, so it carries no subject, time, or evidence.

### Every node must be verifiable

A dataset claim that cannot be re-checked is an opinion. Each node is either
**`observed`** — it must carry re-executable evidence (a query anyone can re-run, see
[Evidence re-execution](#evidence-re-execution)) — or **`latent`** — a real link in
the causal chain that monitoring did not capture; it requires human annotation and is
exempt from recall penalties at evaluation time, so unobservable truth never punishes
the model. Edges carry a verification level for the same reason: `interventional`
(reproduced across repeated injections — the gold standard) or `consistency-checked`
(passed temporal + topological + mechanism checks).

Scenarios also embed **isolated distractors**: real, benign perturbations that are
causally unrelated to the fault. They stay in the graph at degree 0; a model that
wires them into its answer pays the precision penalty. This tests the ability to
reject plausible-but-irrelevant signals, not just to find the true chain.

## Model output contract

`ModelRCAOutput` is deliberately *smaller* than the ground-truth schema — each
asymmetry is a decision about what can be fairly graded:

- **Edges are `(src, dst)` only.** Edge *existence and direction* are objective
  (time order, topology, replay); the *mechanism label* is annotation-side judgment,
  so the model is not asked to produce it.
- **Evidence citation is mandatory** for every node (`hypothesis: true` excepted).
  A hallucinated node cites evidence that does not exist or does not support the
  predicate, and is auto-falsified by re-running the query — the contract makes
  hallucination self-defeating.
- **`hypothesis` nodes** let the model flag unverifiable guesses honestly: they match
  ground-truth `latent` nodes for bonus credit and carry no penalty otherwise.
- **`root_causes` must be stated explicitly**, never inferred from graph sources — a
  missed edge would otherwise fabricate false roots.
- **DAG-ness is not hard-enforced on model output**: a cycle should cost edge scores,
  not reject the whole answer (`find_cycle_nodes()` lets the harness pick its
  degradation policy). Validation rejects only what makes grading impossible.

## Architecture: invariant structural layer + variable vocabulary layer

Testbeds are heterogeneous: a microservice system fails differently from a batch
system, so node predicates, edge mechanisms, and entity types cannot be one global
enum. What *is* common to every system is the shape of the graph and its rules. The
schema therefore splits along exactly that line — structure is code, vocabularies are
per-system config; adding a system means writing one TOML file, never touching code:

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

## Comparing model output to ground truth

`fpg.evaluation` provides a deterministic structural comparison helper for
eval harnesses:

```python
from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth

scenario = Scenario.model_validate_json(open("causal_graph_verified.json").read())
answer = ModelRCAOutput.model_validate_json(raw_model_answer)
comparison = compare_model_to_ground_truth(answer, scenario)

comparison.root_subjects.f1  # root entity match, predicate-agnostic
comparison.subjects.f1       # affected entity match, predicate-agnostic
comparison.soft_subject_edges.f1  # entity edge match with contracted-path decay
comparison.root_nodes.f1     # exact (subject, predicate) root match diagnostic
comparison.nodes.f1          # exact (subject, predicate) node match diagnostic
comparison.soft_edges.f1     # exact edge match with contracted-path decay diagnostic
comparison.subject_path_match_hit  # at least one full entity path matched
comparison.subject_path_reachability_hit # root/symptom entities connected in both graphs
comparison.missing_edges     # readable labels for diagnostics
```

The helper's default scalar score is predicate-agnostic: `0.4 *
root_subject_f1 + 0.3 * subject_f1 + 0.3 * soft_subject_edge_f1`. Predicate-exact
node/edge/path metrics are still reported as diagnostics, but they do not affect
the score. This keeps RCA evaluation focused on whether the model found the
right affected entities and propagation chain even when the ground-truth
predicate label is underspecified or uses a different anomaly taxonomy.

Ground-truth gate nodes are collapsed because `ModelRCAOutput` has no gate node
type. Terminal symptoms are non-gate, non-isolated grounded nodes that have
incoming causes and no outgoing grounded effects. Paths are enumerated from root
causes to those terminal symptoms; reading the same result backward gives the
RCA-style "can this symptom trace back to this root?" reachability check. Path
hits are reported as diagnostics rather than folded into the scalar score. The
helper does not score time overlap or re-execute evidence; those checks can be
layered by a benchmark-specific harness.

## Entity type extension

Entity **types** (prefixes) form an open set, and **all the data lives in profiles**
(fully isomorphic to the vocabulary): each system declares its entity types in its own
profile's `[entity_types.*]` sections (e.g. `svc`/`pod` or `job`/`stage`; `parent`
declares the granularity hierarchy used by scoring decay), which the bundle materializes
into a registry (`schema.entity_registry`). `fpg.entities` provides only the
mechanism (registry + entry point discovery) and carries no data; the entry point
(`[project.entry-points."fpg.entity_types"]`) remains as a supplementary
"install-to-register" channel.

Validation is two-tiered, same as predicates/mechanisms: the structural `EntityRef`
only checks the `<prefix>:<name>` format, while the **profile-bound models constrain
the prefix, enum-like, to the entity types the profile declares** (the factory
generates a `^(svc|pod|…):` pattern for `subject` / `target_entity`; an undeclared
prefix fails validation). The registry remains for granularity lookups
(`ancestors()`, used by scoring decay) and for auditing structural-layer data
(`unregistered_prefixes(refs)`). Entity **instances** (which concrete services exist)
are testbed data and stay out of the schema.

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
| `*.subject` / `target_entity` | `<prefix>:<name>`; in profile-bound models the prefix is constrained, enum-like, to the profile's declared entity types; structural layer checks the format only | `types.EntityRef`; binding: `factory.build_schema` |
| `*.time` | `[start, end]` ISO 8601 interval; instantaneous events have start == end | `types.TimeInterval` |
| `scenario.schema_version` | semver; profile-bound models pin it to exactly the current `SCHEMA_VERSION` | `scenario.Scenario`; binding: `factory.build_schema` |
| `scenario.vocab_version` | `core-<semver>[+<system>-<semver>]`; profile-bound models pin it to exactly the bound profile's `vocab_version` | `profile.VocabProfile`; binding: `factory.build_schema` |
| `injection.fault_type` | Free-form string (value space = the injection tool's fault catalog; no enum enforced) | `scenario.Injection` |

Structural constraints beyond the enums (DAG, preconditions have no in-edges, time
consistency, observed nodes require evidence, isolated distractors have degree 0, etc.)
all live in the structural layer (`fpg.scenario` / `fpg.model_output`) and are inherited
by the factory-generated models; see the `scenario.py` module docstring for the full list.
