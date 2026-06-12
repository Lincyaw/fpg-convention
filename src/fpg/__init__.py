"""Fault Propagation Graph (FPG) schema definitions.

Core package: contains only the structure definitions — the ground-truth
scenario schema and the model output contract. Tooling (validation CLI,
scoring, annotation pipeline) lives in separate packages that depend on
this one.

The schema splits into two halves:

  INVARIANT (code):   node shapes (event/precondition/gate), graph rules,
                      evidence structure, Combine/Grounding/... enums —
                      fpg.scenario and fpg.model_output (structural layer).
  PER-SYSTEM (data):  predicate/mechanism vocabularies and entity types —
                      VocabProfiles loaded from JSON/TOML/YAML config files,
                      bound to concrete pydantic models by
                      fpg.factory.build_schema(). The core profile
                      (fpg/profiles/core.toml) is intentionally empty: it
                      only anchors the version lineage; every system brings
                      its own vocabulary.

The top-level ``Scenario`` / ``ModelRCAOutput`` (and the node/edge classes)
are the STRUCTURAL models: they validate everything except vocabulary
membership (predicate/mechanism are checked as snake_case strings only).
To enforce a system's vocabulary, use
``build_schema(load_profile("config/<system>.toml"))``.

Versioning policy: fpg.version (semconv-style; schema and vocabulary
are versioned independently with semver).

Module layout:
  version       SCHEMA_VERSION and the evolution policy
  vocab         invariant enums + VocabEntry/Stability registry types
  profile       VocabProfile (the variable half), load_profile(),
                CORE_PROFILE (the empty lineage anchor, fpg/profiles/core.toml)
  factory       build_schema(profile) -> SchemaBundle
  entities      extensible entity-type registry (core types + per-testbed
                registration via register()/entry points)
  types         base types: EntityRef / TimeInterval / Evidence(+Query)
  scenario      structural ground-truth schema (vocabulary-agnostic bases)
  model_output  structural model output contract
"""

from .entities import EntityType, EntityTypeRegistry
from .factory import SchemaBundle, build_schema
from .model_output import ModelNode, ModelRCAOutput
from .profile import CORE_PROFILE, EntityTypeSpec, VocabProfile, load_profile
from .scenario import (
    Edge,
    EventNode,
    GateNode,
    Graph,
    PreconditionNode,
    Scenario,
)
from .types import EntityRef, Evidence, EvidenceQuery, TimeInterval
from .version import SCHEMA_VERSION
from .vocab import (
    AnnotationSource,
    Combine,
    Grounding,
    QueryLanguage,
    Stability,
    StructuralRelation,
    VerificationLevel,
    VocabEntry,
)

__all__ = [
    # versioning (vocabulary versions live in profiles: CORE_PROFILE.version)
    "SCHEMA_VERSION",
    # vocab (invariant)
    "Stability",
    "VocabEntry",
    "QueryLanguage",
    "Grounding",
    "Combine",
    "AnnotationSource",
    "VerificationLevel",
    "StructuralRelation",
    # profile + factory (vocabularies are per-system config; core is empty)
    "VocabProfile",
    "EntityTypeSpec",
    "CORE_PROFILE",
    "load_profile",
    "build_schema",
    "SchemaBundle",
    # entities (registry mechanism; data lives in profiles)
    "EntityType",
    "EntityTypeRegistry",
    # types
    "EntityRef",
    "TimeInterval",
    "Evidence",
    "EvidenceQuery",
    # structural ground truth models (vocabulary-agnostic)
    "EventNode",
    "PreconditionNode",
    "GateNode",
    "Edge",
    "Graph",
    "Scenario",
    # structural model output contract (vocabulary-agnostic)
    "ModelNode",
    "ModelRCAOutput",
]
