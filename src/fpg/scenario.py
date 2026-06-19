"""Ground-truth scenario file schema.

STRUCTURAL LAYER: the models here are the invariant half of the schema —
node shapes, graph rules, evidence structure. They validate ``predicate``
and ``mechanism`` only as snake_case strings. To enforce a system's
vocabulary, bind a profile via fpg.factory.build_schema(profile), which
derives subclasses with those fields constrained to the profile's enums.

Nodes form a discriminated union on ``kind`` — the three kinds have
genuinely different shapes, so each gets its own class instead of one flat
class full of conditionally-required fields:

  - EventNode:     something anomalous happened on a real entity.
                   Grounded (subject/predicate/time/evidence), may have
                   incoming edges.
  - PreconditionNode: a standing weakness that existed BEFORE this fault
                   (e.g. undersized pool) and enabled it. Grounded like an
                   event, but never caused by this fault, so it has no
                   incoming edges (and hence no ``combine``).
  - GateNode:      pure boolean wiring, not a real thing in the system.
                   Exists only to express mixed combinations like
                   (A AND B) OR C that a single per-node ``combine`` cannot.
                   No subject/predicate/time/evidence.

Remaining cross-field constraints (enforced by validators):
  Node level
    - grounding == observed -> evidence non-empty
    - predicate == other -> description required
  Edge level
    - src != dst; mechanism == other -> description required
  Graph level
    - node ids unique; edge endpoints exist
    - DAG (no cycles)
    - precondition nodes have no incoming edges
    - event nodes with in-degree >= 2 must carry combine
    - gate nodes have in-degree >= 2 and out-degree >= 1
    - temporal consistency: src.time.start <= dst.time.start
      (skipped when either endpoint is a gate, which has no time)
    - isolated_distractor nodes have degree 0
    - injection source node_id exists and has no incoming edges
"""

from collections import defaultdict, deque
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from .types import EntityRef, Evidence, TimeInterval
from .version import SCHEMA_VERSION
from .vocab import (
    AnnotationSource,
    Combine,
    Grounding,
    VerificationLevel,
)


class _GroundedNode(BaseModel):
    """Shared shape of nodes that describe a real entity (event/precondition)."""

    id: str = Field(min_length=1, description="Node id, unique within the scenario")
    subject: EntityRef
    predicate: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Failure mode. This structural layer only enforces the "
        "snake_case format; vocabulary membership is enforced by the "
        "profile-bound models from fpg.factory.build_schema(). 'other' "
        "requires description",
    )
    time: TimeInterval = Field(
        description="Anomaly time interval; instantaneous events use a "
        "degenerate interval (start == end)"
    )
    grounding: Literal[Grounding.OBSERVED, Grounding.LATENT] = Field(
        description="observed (mechanically re-verifiable, evidence required) "
        "| latent (unmonitored, human-annotated, recall-exempt at evaluation)"
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Re-executable evidence; must be non-empty when grounding=observed",
    )
    annotation: AnnotationSource | None = Field(
        default=None,
        description="Annotation provenance: auto | human | replay-verified",
    )
    isolated_distractor: bool = Field(
        default=False,
        description="Benign-perturbation node: real but causally unrelated "
        "to this fault; must stay isolated (degree 0). A model wiring it "
        "into the propagation graph pays the edge-precision penalty",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable note, not scored; required when predicate=other",
    )

    @model_validator(mode="after")
    def _check_grounded(self) -> "_GroundedNode":
        if self.grounding is Grounding.OBSERVED and not self.evidence:
            raise ValueError(
                f"node {self.id!r}: grounding=observed requires non-empty evidence"
            )
        if self.predicate == "other" and not self.description:
            raise ValueError(f"node {self.id!r}: predicate=other requires description")
        return self


class EventNode(_GroundedNode):
    """Anomalous event/state on a real entity; may have incoming edges."""

    kind: Literal["event"] = "event"
    combine: Combine | None = Field(
        default=None,
        description="Combination semantics over ALL incoming edges, AND|OR; "
        "required when in-degree >= 2 (graph-level check)",
    )


class PreconditionNode(_GroundedNode):
    """Standing weakness that existed BEFORE this fault and enabled it
    (e.g. undersized pool, missing timeout). The distinguishing axis vs
    EventNode is causal origin, not duration: a precondition was already
    there when the fault was injected, so it never has incoming edges —
    and needs no ``combine``. Typically combined with a trigger event via
    AND on the effect node ("trigger AND weakness").
    """

    kind: Literal["precondition"] = "precondition"


class GateNode(BaseModel):
    """Pure boolean connector for mixed combinations like (A AND B) OR C.

    Not a real thing in the system: no subject, no predicate, no time, no
    evidence. Wire the sub-causes into the gate, then the gate (plus the
    remaining causes) into the effect node.
    """

    kind: Literal["gate"] = "gate"
    id: str = Field(min_length=1, description="Node id, unique within the scenario")
    combine: Combine = Field(
        description="The gate's boolean operator over its incoming edges, AND|OR"
    )
    description: str | None = Field(default=None, description="Not scored")


Node = Annotated[EventNode | PreconditionNode | GateNode, Field(discriminator="kind")]


class Edge(BaseModel):
    """Causal edge: asserts that src (partially) caused dst."""

    src: str = Field(description="Cause-side node id")
    dst: str = Field(description="Effect-side node id")
    mechanism: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Propagation mechanism. This structural layer only "
        "enforces the snake_case format; vocabulary membership is enforced "
        "by the profile-bound models from fpg.factory.build_schema(). "
        "'other' requires description",
    )
    verification: VerificationLevel = Field(
        description="interventional (main chain, replay-verified) | "
        "consistency-checked (side branch, triple check passed)"
    )
    description: str | None = Field(default=None, description="Not scored")

    @model_validator(mode="after")
    def _check_edge(self) -> "Edge":
        if self.src == self.dst:
            raise ValueError(
                f"edge {self.src!r} -> {self.dst!r}: self-loop is not allowed"
            )
        if self.mechanism == "other" and not self.description:
            raise ValueError(
                f"edge {self.src!r} -> {self.dst!r}: "
                f"mechanism=other requires description"
            )
        return self


class Injection(BaseModel):
    """Injection record: anchors a root-cause source node of the graph to an
    injection experiment.
    """

    node_id: str = Field(
        description="Id of the injection source node in the graph "
        "(must have no incoming edges)"
    )
    fault_type: str = Field(
        min_length=1,
        description="Fault type identifier of the injection tool. The value "
        "space is the tool's fault catalog (e.g. ChaosMesh/ChaosBlade fault "
        "ids); intentionally not an enum",
    )
    target_entity: EntityRef
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Injection parameters"
    )
    time: TimeInterval = Field(description="Injection active interval")
    replay_count: int = Field(
        default=0,
        ge=0,
        description="Number of replay verifications; basis for marking "
        "main-chain edges as interventional (>= 2 recommended)",
    )


class Graph(BaseModel):
    """Propagation graph: a DAG. Time anchors break cycles by construction;
    feedback loops are unrolled over time (the same (subject, predicate)
    recurring later becomes a new node, never a back-edge).
    """

    nodes: list[Node] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph(self) -> "Graph":
        by_id = {n.id: n for n in self.nodes}
        if len(by_id) != len(self.nodes):
            seen: set[str] = set()
            dup = ""
            for node in self.nodes:
                if node.id in seen:
                    dup = node.id
                    break
                seen.add(node.id)
            raise ValueError(f"duplicate node id {dup!r}")

        in_deg: dict[str, int] = defaultdict(int)
        out_deg: dict[str, int] = defaultdict(int)
        succ: dict[str, list[str]] = defaultdict(list)
        for e in self.edges:
            for endpoint in (e.src, e.dst):
                if endpoint not in by_id:
                    raise ValueError(f"edge references unknown node {endpoint!r}")
            in_deg[e.dst] += 1
            out_deg[e.src] += 1
            succ[e.src].append(e.dst)

        # Kahn topological sort for cycle detection
        deg = {nid: in_deg[nid] for nid in by_id}
        queue = deque(nid for nid, d in deg.items() if d == 0)
        visited = 0
        while queue:
            nid = queue.popleft()
            visited += 1
            for nxt in succ[nid]:
                deg[nxt] -= 1
                if deg[nxt] == 0:
                    queue.append(nxt)
        if visited != len(by_id):
            cyclic = sorted(nid for nid, d in deg.items() if d > 0)
            raise ValueError(f"graph contains a cycle involving nodes {cyclic}")

        for n in self.nodes:
            if isinstance(n, PreconditionNode) and in_deg[n.id] > 0:
                raise ValueError(
                    f"precondition node {n.id!r} must not have incoming edges"
                )
            if isinstance(n, EventNode) and in_deg[n.id] >= 2 and n.combine is None:
                raise ValueError(
                    f"node {n.id!r} has in-degree {in_deg[n.id]} and requires "
                    f"combine (AND|OR)"
                )
            if isinstance(n, GateNode):
                if in_deg[n.id] < 2 or out_deg[n.id] < 1:
                    raise ValueError(
                        f"gate node {n.id!r} must have in-degree >= 2 and "
                        f"out-degree >= 1 (got in={in_deg[n.id]}, out={out_deg[n.id]})"
                    )
            elif n.isolated_distractor and (in_deg[n.id] or out_deg[n.id]):
                raise ValueError(
                    f"isolated_distractor node {n.id!r} must have degree 0"
                )

        for e in self.edges:
            src, dst = by_id[e.src], by_id[e.dst]
            if isinstance(src, GateNode) or isinstance(dst, GateNode):
                continue  # gates have no time; consistency holds transitively
            if src.time.start > dst.time.start:
                raise ValueError(
                    f"edge {e.src!r} -> {e.dst!r} violates temporal consistency: "
                    f"cause starts at {src.time.start}, effect starts at "
                    f"{dst.time.start}"
                )
        return self

    def node(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)

    @property
    def root_causes(self) -> list[Node]:
        """Root causes = nodes with no incoming edges (injection points and
        preconditions), excluding isolated distractor nodes. Gates never
        appear here (they always have incoming edges).
        """
        targets = {e.dst for e in self.edges}
        return [
            n
            for n in self.nodes
            if n.id not in targets
            and not (isinstance(n, _GroundedNode) and n.isolated_distractor)
        ]


class Scenario(BaseModel):
    """Fault scenario file (ground truth). Self-describing:
    carries the schema and vocabulary versions it conforms to (see
    fpg.version for the evolution policy).
    """

    schema_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Structural schema version (semver) this file conforms "
        f"to; current: {SCHEMA_VERSION}",
    )
    scenario_id: str = Field(min_length=1)
    testbed: str = Field(min_length=1, description="Testbed identifier")
    vocab_version: str = Field(
        pattern=r"^[a-z][a-z0-9_]*-\d+\.\d+\.\d+(?:\+[a-z][a-z0-9_]*-\d+\.\d+\.\d+)*$",
        description="Vocabulary version declaration: "
        "'core-<semver>[+<extension>-<semver>]', e.g. 'core-0.1.0+ecom-0.1.0'. "
        "Profile-bound models pin this to exactly the bound profile's "
        "vocab_version",
    )
    segmentation_gap_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Node segmentation gap parameter g: anomalies of the "
        "same (subject, predicate) merge into one node when the normal gap "
        "between them is < g, and split otherwise",
    )
    injections: list[Injection] = Field(
        min_length=1,
        description="Injection records; multi-fault scenarios carry several",
    )
    graph: Graph

    @model_validator(mode="after")
    def _check_scenario(self) -> "Scenario":
        ids = {n.id for n in self.graph.nodes}
        targets = {e.dst for e in self.graph.edges}
        for inj in self.injections:
            if inj.node_id not in ids:
                raise ValueError(f"injection references unknown node {inj.node_id!r}")
            if inj.node_id in targets:
                raise ValueError(
                    f"injection node {inj.node_id!r} must be a root (no incoming edges)"
                )
        return self
