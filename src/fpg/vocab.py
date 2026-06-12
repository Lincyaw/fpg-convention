"""Invariant vocabularies.

Only the schema's INVARIANT value spaces live here as true enums —
Grounding, Combine, AnnotationSource, VerificationLevel, StructuralRelation,
QueryLanguage — plus the VocabEntry/Stability registry types.

PER-SYSTEM vocabulary data (node predicates, edge mechanisms, entity types)
is configuration, not code: every system defines its own vocabulary in a
profile config file (see fpg.profile and config/template.toml in the
repository) and bakes it into concrete pydantic models via
fpg.factory.build_schema(). The core profile is intentionally empty — it
only anchors the version lineage. The structural models in fpg.scenario
validate these fields only as snake_case strings; vocabulary membership is
enforced by the profile-bound models the factory generates.

Naming conventions:
  - Values are lowercase ``snake_case``.
  - Node predicates are noun phrases naming a *failure mode* of the subject
    entity (what is broken), not its cause or consequence.
  - Edge mechanisms are noun phrases naming a *propagation channel behavior*
    (how badness transfers between entities).

Governance:
  - ``other`` is the escape hatch: the factory injects it into every
    generated enum; using it requires a free-text ``description`` on the
    carrying object. When one recurring OTHER pattern exceeds ~5% of
    annotations, promote it to a new atomic entry (MINOR version bump).
  - Enumeration method: Cartesian expansion over interaction channels
    (sync call / async messaging / shared resource / control plane / data
    dependency) x resource types, rather than ad-hoc listing.

Versioning: see fpg.version (structure) and each profile's ``version``
field (vocabularies). Entry lifecycle: experimental -> stable, or
experimental/stable -> deprecated (with ``renamed_to``) -> removed at the
next MAJOR release.
"""

from dataclasses import dataclass
from enum import Enum


class Stability(str, Enum):
    """Lifecycle stability of a vocabulary entry (semconv-style)."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class VocabEntry:
    """Registry metadata for one vocabulary value."""

    brief: str
    stability: Stability = Stability.EXPERIMENTAL
    since: str = "0.1.0"
    renamed_to: str | None = None  # set iff stability == DEPRECATED


class QueryLanguage(str, Enum):
    """Query language of an evidence statement; tells the re-verifier how
    to execute it. Closed enum, extended as new testbeds need new languages
    (each addition is a MINOR schema bump).
    """

    SQL = "sql"
    PROMQL = "promql"
    LOGQL = "logql"
    HTTP = "http"
    # Escape hatch: a language outside the vocabulary; explain in the
    # carrying Evidence's explanation
    OTHER = "other"


# Node kinds (event | precondition | gate) are not an enum here: they are the
# discriminator of the Node union in fpg.scenario, and each kind has its own
# class (EventNode / PreconditionNode / GateNode) defining its exact shape.


class Grounding(str, Enum):
    """Node verification basis, for grounded nodes (event/precondition).

    - ``observed``: has evidence, mechanically re-verifiable
      (evidence required);
    - ``latent``: not covered by monitoring; human annotation mandatory;
      exempt from recall penalty at evaluation time.

    Gate nodes carry no grounding at all — they are pure boolean wiring,
    not statements about the system.
    """

    OBSERVED = "observed"
    LATENT = "latent"


class Combine(str, Enum):
    """Multi-parent combination semantics; required when in-degree >= 2.

    - ``AND``: all parents jointly necessary (typical: trigger AND weakness);
    - ``OR``: any single parent suffices.
    Mixed boolean expressions use gate nodes.
    """

    AND = "AND"
    OR = "OR"


class AnnotationSource(str, Enum):
    """Annotation provenance: how a node/edge label was produced."""

    AUTO = "auto"
    HUMAN = "human"
    REPLAY_VERIFIED = "replay-verified"


class VerificationLevel(str, Enum):
    """Edge verification strength.

    - ``interventional``: main-chain edge, reproduced across repeated
      injection experiments (interventional gold standard);
    - ``consistency-checked``: side edge, passed the temporal + topological
      + mechanism triple check.
    """

    INTERVENTIONAL = "interventional"
    CONSISTENCY_CHECKED = "consistency-checked"


class StructuralRelation(str, Enum):
    """Topology-layer structural relations between entities (the static
    deployment/dependency topology, distinct from causal edges).
    """

    CALLS = "calls"
    RUNS_ON = "runs-on"
    CONNECTS_TO = "connects-to"
    DEPENDS_ON = "depends-on"
