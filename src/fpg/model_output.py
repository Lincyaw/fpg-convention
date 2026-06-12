"""Model output contract: the structure an evaluated model must emit.

Differences from the ground-truth schema (the trade-offs settled in design):
  - Edges carry only (src, dst) existence and direction — no mechanism (its
    labeling subjectivity stays on the annotation side), no evidence (an
    edge's evidence lives in its endpoint nodes), no combine;
  - Nodes must cite evidence (hallucinated nodes are auto-falsified by the
    re-verifiers), except hypothesis=True guess nodes — these correspond to
    ground-truth latent nodes and earn bonus credit when matched;
  - root_causes must be stated explicitly, never inferred from graph
    sources (a missed edge would fabricate false sources).

Validation strength: this contract enforces referential integrity and the
evidence rule only. DAG-ness is intentionally NOT a hard constraint — when a
model emits a cycle the evaluation should score edges individually rather
than reject the whole output; ``find_cycle_nodes()`` is provided so the
harness can choose its own degradation policy.
"""

from collections import defaultdict, deque

from pydantic import BaseModel, Field, model_validator

from .types import EntityRef, Evidence, TimeInterval


class ModelNode(BaseModel):
    """Propagation-graph node emitted by the evaluated model."""

    id: str = Field(min_length=1, description="Node id, unique within the output")
    subject: EntityRef
    predicate: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Must come from the node vocabulary handed to the "
        "model. This structural layer only enforces the snake_case format; "
        "membership is enforced by the profile-bound models from "
        "fpg.factory.build_schema()",
    )
    time: TimeInterval = Field(
        description="Anomaly interval as judged by the model; aligned via "
        "IoU at evaluation time, endpoint precision not required"
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Mandatory citation (non-empty unless hypothesis=true): "
        "must point at observation data that exists and supports the "
        "predicate; nodes citing nonexistent or non-supporting evidence are "
        "auto-falsified by the re-verifiers",
    )
    hypothesis: bool = Field(
        default=False,
        description="Guess-node marker: exempt from evidence; earns bonus "
        "credit when it matches a ground-truth latent node, no penalty "
        "otherwise",
    )

    @model_validator(mode="after")
    def _check_evidence_contract(self) -> "ModelNode":
        if not self.hypothesis and not self.evidence:
            raise ValueError(
                f"model node {self.id!r}: evidence is mandatory unless hypothesis=true"
            )
        return self


class ModelEdge(BaseModel):
    """Causal edge emitted by the model: existence and direction only."""

    src: str = Field(description="Cause-side node id")
    dst: str = Field(description="Effect-side node id")

    @model_validator(mode="after")
    def _no_self_loop(self) -> "ModelEdge":
        if self.src == self.dst:
            raise ValueError(f"model edge {self.src!r} -> {self.dst!r}: self-loop")
        return self


class ModelRCAOutput(BaseModel):
    nodes: list[ModelNode] = Field(min_length=1)
    edges: list[ModelEdge] = Field(default_factory=list)
    root_causes: list[str] = Field(
        min_length=1,
        description="Explicitly nominated root-cause node ids (possibly "
        "several: injection points plus preconditions), ordered by "
        "descending confidence; RCA@k takes the first k",
    )

    @model_validator(mode="after")
    def _check_refs(self) -> "ModelRCAOutput":
        ids = {n.id: None for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate model node id")
        for e in self.edges:
            for endpoint in (e.src, e.dst):
                if endpoint not in ids:
                    raise ValueError(f"model edge references unknown node {endpoint!r}")
        for rc in self.root_causes:
            if rc not in ids:
                raise ValueError(f"root_causes references unknown node {rc!r}")
        return self

    def find_cycle_nodes(self) -> list[str]:
        """Return ids of nodes on a cycle (empty list = DAG). The evaluation
        harness decides the degradation policy.
        """
        in_deg: dict[str, int] = {n.id: 0 for n in self.nodes}
        succ: dict[str, list[str]] = defaultdict(list)
        for e in self.edges:
            in_deg[e.dst] += 1
            succ[e.src].append(e.dst)
        queue = deque(nid for nid, d in in_deg.items() if d == 0)
        while queue:
            nid = queue.popleft()
            for nxt in succ[nid]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        return sorted(nid for nid, d in in_deg.items() if d > 0)
