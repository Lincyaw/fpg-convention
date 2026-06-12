"""Factory: bind a VocabProfile to concrete pydantic models.

``build_schema(profile)`` derives, from the structural base models in
fpg.scenario / fpg.model_output, a per-system set of models whose
``predicate`` / ``mechanism`` fields are constrained to the profile's
vocabulary (as dynamically created str-enums, with ``other`` injected as
the escape hatch). The result is a ``SchemaBundle``:

    profile = load_profile("config/<system>.toml")
    schema = build_schema(profile)
    scenario = schema.Scenario.model_validate_json(raw)   # vocab enforced
    schema.ModelRCAOutput.model_json_schema()             # for the LLM

Generated classes inherit the structural bases, so every graph-level
validator is inherited and ``isinstance(x, fpg.scenario.EventNode)`` keeps
working across profiles. The bundle also carries an entity-type registry
preloaded with core types plus the profile's extension types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, cast

from pydantic import Field, create_model

from . import model_output as mo
from . import scenario as sc
from .entities import EntityTypeRegistry
from .profile import VocabProfile

OTHER_VALUE = "other"


def _make_enum(class_name: str, values: tuple[str, ...]) -> type[Enum]:
    members = {v.upper(): v for v in (*values, OTHER_VALUE)}
    # The functional Enum API returns the new enum class; pyright's stubs
    # type it as an instance, hence the cast (ty deems it redundant).
    return cast(type[Enum], Enum(class_name, members, type=str))  # ty: ignore[redundant-cast]


@dataclass(frozen=True)
class SchemaBundle:
    """All artifacts of one profile-bound schema."""

    profile: VocabProfile
    NodePredicate: type[Enum]
    EdgeMechanism: type[Enum]
    # ground truth
    EventNode: type[sc.EventNode]
    PreconditionNode: type[sc.PreconditionNode]
    GateNode: type[sc.GateNode]
    Edge: type[sc.Edge]
    Graph: type[sc.Graph]
    Scenario: type[sc.Scenario]
    # model output contract
    ModelNode: type[mo.ModelNode]
    ModelRCAOutput: type[mo.ModelRCAOutput]
    # registry materialized from the profile's full entity-type set
    entity_registry: EntityTypeRegistry

    def vocab_for_model(self) -> dict[str, str]:
        """Predicate vocabulary handed to the evaluated model as its
        allowed node value space: value -> brief.
        """
        return {
            value: entry.brief for value, entry in self.profile.node_predicates.items()
        }


def build_schema(profile: VocabProfile) -> SchemaBundle:
    """Generate the profile-bound pydantic models for one system."""
    node_predicate = _make_enum("NodePredicate", tuple(profile.node_predicates))
    edge_mechanism = _make_enum("EdgeMechanism", tuple(profile.edge_mechanisms))

    def predicate_field() -> tuple:
        return (
            node_predicate,
            Field(
                description=f"Failure mode from the {profile.vocab_version!r} "
                "vocabulary; 'other' requires description"
            ),
        )

    event_node = create_model(
        "EventNode", __base__=sc.EventNode, predicate=predicate_field()
    )
    precondition_node = create_model(
        "PreconditionNode", __base__=sc.PreconditionNode, predicate=predicate_field()
    )
    # Dynamic type expressions below are evaluated at runtime by pydantic;
    # they are values here, not static annotations.
    node_union: Any = Annotated[
        event_node | precondition_node | sc.GateNode,  # ty: ignore[invalid-type-form]
        Field(discriminator="kind"),
    ]
    edge = create_model(
        "Edge",
        __base__=sc.Edge,
        mechanism=(
            edge_mechanism,
            Field(
                description=f"Propagation mechanism from the "
                f"{profile.vocab_version!r} vocabulary; 'other' requires "
                "description"
            ),
        ),
    )
    graph = create_model(
        "Graph",
        __base__=sc.Graph,
        nodes=(list[node_union], Field(min_length=1)),
        edges=(list[edge], Field(default_factory=list)),  # ty: ignore[invalid-type-form]
    )
    scenario = create_model("Scenario", __base__=sc.Scenario, graph=(graph, ...))

    model_node = create_model(
        "ModelNode", __base__=mo.ModelNode, predicate=predicate_field()
    )
    model_output = create_model(
        "ModelRCAOutput",
        __base__=mo.ModelRCAOutput,
        nodes=(list[model_node], Field(min_length=1)),  # ty: ignore[invalid-type-form]
    )

    registry = EntityTypeRegistry(profile.entity_type_objects())

    return SchemaBundle(
        profile=profile,
        NodePredicate=node_predicate,
        EdgeMechanism=edge_mechanism,
        EventNode=event_node,
        PreconditionNode=precondition_node,
        GateNode=sc.GateNode,
        Edge=edge,
        Graph=graph,
        Scenario=scenario,
        ModelNode=model_node,
        ModelRCAOutput=model_output,
        entity_registry=registry,
    )
