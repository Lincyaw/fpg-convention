"""Factory: bind a VocabProfile to concrete pydantic models.

``build_schema(profile)`` derives, from the structural base models in
fpg.scenario / fpg.model_output, a per-system set of models whose
``predicate`` / ``mechanism`` fields are constrained to the profile's
vocabulary (as dynamically created str-enums, with ``other`` injected as
the escape hatch) and whose entity references (``subject`` /
``target_entity``) constrain the prefix to the profile's declared entity
types. The result is a ``SchemaBundle``:

    profile = load_profile("config/<system>.toml")
    schema = build_schema(profile)
    scenario = schema.Scenario.model_validate_json(raw)   # vocab enforced
    schema.ModelRCAOutput.model_json_schema()             # for the LLM

Generated classes inherit the structural bases, so every graph-level
validator is inherited and ``isinstance(x, fpg.scenario.EventNode)`` keeps
working across profiles. The bundle also carries an entity-type registry
preloaded with core types plus the profile's extension types.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, cast

from pydantic import Field, create_model

from . import model_output as mo
from . import scenario as sc
from .entities import EntityTypeRegistry
from .profile import VocabProfile
from .version import SCHEMA_VERSION

OTHER_VALUE = "other"


def _make_enum(class_name: str, values: tuple[str, ...]) -> type[Enum]:
    members = {v.upper(): v for v in (*values, OTHER_VALUE)}
    # The functional Enum API returns the new enum class; pyright's stubs
    # type it as an instance, hence the cast (ty deems it redundant).
    return cast(type[Enum], Enum(class_name, members, type=str))  # ty: ignore[redundant-cast]


def _make_entity_ref(profile: VocabProfile) -> Any:
    """Entity-reference type with the prefix constrained, enum-like, to the
    entity types declared in the profile.
    """
    prefixes = tuple(profile.entity_types)
    if not prefixes:
        raise ValueError(
            f"profile {profile.name!r} declares no entity_types; a system "
            f"profile must define its full entity-type set before models "
            f"can be generated"
        )
    alternation = "|".join(re.escape(p) for p in prefixes)
    return Annotated[
        str,
        Field(
            pattern=rf"^(?:{alternation}):[A-Za-z0-9][A-Za-z0-9._>-]*$",
            description=f"Entity reference '<prefix>:<name>'; prefix must be "
            f"one of the {profile.vocab_version!r} entity types: "
            f"{', '.join(prefixes)}",
            examples=[f"{prefixes[0]}:example-instance"],
        ),
    ]


@dataclass(frozen=True)
class SchemaBundle:
    """All artifacts of one profile-bound schema."""

    profile: VocabProfile
    NodePredicate: type[Enum]
    EdgeMechanism: type[Enum]
    # entity reference with the prefix constrained to the profile's types
    # (an Annotated[str, ...] type, not a class)
    EntityRef: Any
    # ground truth
    EventNode: type[sc.EventNode]
    PreconditionNode: type[sc.PreconditionNode]
    GateNode: type[sc.GateNode]
    Edge: type[sc.Edge]
    Injection: type[sc.Injection]
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
    entity_ref = _make_entity_ref(profile)

    def predicate_field() -> tuple:
        return (
            node_predicate,
            Field(
                description=f"Failure mode from the {profile.vocab_version!r} "
                "vocabulary; 'other' requires description"
            ),
        )

    event_node = create_model(
        "EventNode",
        __base__=sc.EventNode,
        subject=(entity_ref, ...),
        predicate=predicate_field(),
    )
    precondition_node = create_model(
        "PreconditionNode",
        __base__=sc.PreconditionNode,
        subject=(entity_ref, ...),
        predicate=predicate_field(),
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
    injection = create_model(
        "Injection", __base__=sc.Injection, target_entity=(entity_ref, ...)
    )
    # Self-description is pinned, not free-form: a file validated by this
    # bundle must declare exactly the versions the bundle was built from.
    scenario = create_model(
        "Scenario",
        __base__=sc.Scenario,
        schema_version=(
            str,
            Field(
                pattern=rf"^{re.escape(SCHEMA_VERSION)}$",
                description=f"Must be {SCHEMA_VERSION!r}, the structural "
                "schema version this bundle was generated by",
            ),
        ),
        vocab_version=(
            str,
            Field(
                pattern=rf"^{re.escape(profile.vocab_version)}$",
                description=f"Must be {profile.vocab_version!r}, the "
                "vocabulary this bundle binds",
            ),
        ),
        injections=(list[injection], Field(min_length=1)),  # ty: ignore[invalid-type-form]
        graph=(graph, ...),
    )

    model_node = create_model(
        "ModelNode",
        __base__=mo.ModelNode,
        subject=(entity_ref, ...),
        predicate=predicate_field(),
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
        EntityRef=entity_ref,
        EventNode=event_node,
        PreconditionNode=precondition_node,
        GateNode=sc.GateNode,
        Edge=edge,
        Injection=injection,
        Graph=graph,
        Scenario=scenario,
        ModelNode=model_node,
        ModelRCAOutput=model_output,
        entity_registry=registry,
    )
