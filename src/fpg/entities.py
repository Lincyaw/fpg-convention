"""Entity-type registry MECHANISM.

This module is pure mechanism; the DATA (which entity types exist) lives in
each system's vocabulary profile (fpg.profile), because entity kinds are
per-system — exactly like predicates and mechanisms.

Distinguish two layers:
  - Entity *types* (prefixes such as ``svc``, ``pod``, ``func``):
    schema-level, declared in a profile and materialized into a registry by
    fpg.factory.build_schema().
  - Entity *instances* (the catalog: which services actually exist): plain
    per-testbed data, never part of the schema.

Ways to bring extension types in, in order of preference:
  1. Profile config file: declare them under ``[entity_types.*]`` —
     they ride along with the system's vocabulary (see fpg.profile).
  2. Entry points (automatic): a testbed package declares

         [project.entry-points."fpg.entity_types"]
         my_testbed = "my_testbed.entities:ENTITY_TYPES"

     where the target is an iterable of EntityType (or a zero-arg callable
     returning one). ``registry.load_entry_points()`` discovers and registers
     everything installed in the environment.
  3. Programmatic: ``registry.register(EntityType(...))`` or
     ``registry.register_extension("ecom", [...])``.

Validation contract: the structural EntityRef pattern (fpg.types) accepts any
well-formed "<prefix>:<name>"; whether a prefix is *registered* is a separate
strict check (``unregistered_prefixes``) run by downstream tooling, so that
core schema validation never depends on which extensions happen to be
installed.

Entity types follow the same lifecycle as vocabulary entries (fpg.version):
new types start ``experimental``; renames go through ``deprecated``.
"""

import re
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Iterable, Iterator, cast

from .vocab import Stability

ENTRY_POINT_GROUP = "fpg.entity_types"

_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class EntityType:
    """One registered entity-type prefix."""

    prefix: str
    brief: str
    stability: Stability = Stability.EXPERIMENTAL
    since: str = "0.1.0"
    # Prefix of the next coarser-grained type (container -> pod -> svc -> ns);
    # consumed by granularity-decay scoring.
    parent: str | None = None
    # Name of the profile (or extension) that declared this type.
    namespace: str = "core"

    def __post_init__(self) -> None:
        if not _PREFIX_PATTERN.match(self.prefix):
            raise ValueError(
                f"invalid entity-type prefix {self.prefix!r}: "
                f"must match {_PREFIX_PATTERN.pattern}"
            )


class EntityTypeRegistry:
    """Registry of entity-type prefixes: core set plus testbed extensions."""

    def __init__(self, types: Iterable[EntityType] = ()) -> None:
        self._types: dict[str, EntityType] = {}
        for et in types:
            self.register(et)

    def register(
        self, entity_type: EntityType, *, replace_existing: bool = False
    ) -> EntityType:
        existing = self._types.get(entity_type.prefix)
        if existing is not None and not replace_existing:
            raise ValueError(
                f"entity-type prefix {entity_type.prefix!r} already registered "
                f"by namespace {existing.namespace!r}; pass replace_existing=True "
                f"to override"
            )
        self._types[entity_type.prefix] = entity_type
        return entity_type

    def register_extension(
        self, namespace: str, types: Iterable[EntityType]
    ) -> list[EntityType]:
        """Register a testbed extension, stamping every type with its namespace."""
        return [self.register(replace(et, namespace=namespace)) for et in types]

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> int:
        """Discover and register extension types installed in the environment.

        Each entry point resolves to an iterable of EntityType or a zero-arg
        callable returning one. Returns the number of types registered.
        Already-registered prefixes raise, surfacing extension conflicts early.
        """
        count = 0
        for ep in metadata.entry_points(group=group):
            target = ep.load()
            types = cast(Iterable[EntityType], target() if callable(target) else target)
            count += len(self.register_extension(ep.name, types))
        return count

    def __contains__(self, prefix: str) -> bool:
        return prefix in self._types

    def __iter__(self) -> Iterator[EntityType]:
        return iter(self._types.values())

    def __len__(self) -> int:
        return len(self._types)

    def get(self, prefix: str) -> EntityType | None:
        return self._types.get(prefix)

    @property
    def prefixes(self) -> list[str]:
        return sorted(self._types)

    @staticmethod
    def prefix_of(ref: str) -> str:
        """Extract the type prefix from an entity reference '<prefix>:<name>'."""
        return ref.split(":", 1)[0]

    def ancestors(self, prefix: str) -> list[str]:
        """Coarser-grained prefixes along the parent chain, nearest first."""
        chain: list[str] = []
        seen = {prefix}
        current = self._types.get(prefix)
        while current is not None and current.parent is not None:
            if current.parent in seen:
                raise ValueError(
                    f"parent cycle in entity hierarchy at {current.parent!r}"
                )
            chain.append(current.parent)
            seen.add(current.parent)
            current = self._types.get(current.parent)
        return chain

    def unregistered_prefixes(self, refs: Iterable[str]) -> list[str]:
        """Strict-layer check: prefixes used by refs but not registered."""
        return sorted({self.prefix_of(r) for r in refs} - set(self._types))


# Entity-type DATA lives in each system's vocabulary profile, not here:
# which entity kinds exist is per-system data, exactly like predicates and
# mechanisms. This module only provides the mechanism (EntityType, the
# registry, and entry-point discovery); fpg.factory.build_schema()
# materializes a registry from a profile.
