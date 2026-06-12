"""Per-system vocabulary profiles: the variable half of the schema.

The schema splits into an invariant structural layer (node shapes, graph
rules, evidence structure — fpg.scenario / fpg.model_output) and a variable
vocabulary layer that differs per system. A ``VocabProfile`` carries the
variable layer for one system:

  - node predicates  (a FaaS system has cold_start, a batch system job_failed)
  - edge mechanisms
  - extension entity types

Profiles compose: a system profile typically extends the core profile
(``CORE_PROFILE``), adding its own entries; duplicate values are an error,
never a silent override. Profiles can be authored in code (they are plain
pydantic objects) or loaded from config files via ``load_profile()``
(JSON / TOML / YAML) — see config/template.toml in
the repository for all available fields:

    # config/bigdata.toml
    name = "bigdata"
    version = "0.1.0"
    extends = "core"

    [node_predicates.job_failed]
    brief = "Batch job terminated with failure"

    [entity_types.job]
    brief = "Batch job"

Binding a profile to concrete pydantic models is the factory's job:
``fpg.factory.build_schema(profile)``.
"""

import json
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import yaml  # ty: ignore[unresolved-import]  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .entities import EntityType
from .vocab import Stability, VocabEntry

_VALUE_PATTERN = r"^[a-z][a-z0-9_]*$"

# 'other' is reserved: the factory injects it into every generated enum
RESERVED_VALUES = frozenset({"other"})


class EntityTypeSpec(BaseModel):
    """Declaration of one extension entity type inside a profile."""

    model_config = ConfigDict(frozen=True)

    brief: str
    stability: Stability = Stability.EXPERIMENTAL
    since: str = "0.1.0"
    parent: str | None = Field(
        default=None,
        description="Prefix of the next coarser-grained type, for "
        "granularity-decay scoring",
    )


class VocabProfile(BaseModel):
    """The complete variable layer for one system."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        pattern=_VALUE_PATTERN, description="Profile name, e.g. 'core', 'ecom'"
    )
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$", description="Profile semver")
    lineage: list[str] = Field(
        default_factory=list,
        description="Versions of the profiles this one extends, in order, "
        "e.g. ['core-0.1.0']",
    )
    node_predicates: dict[str, VocabEntry] = Field(
        default_factory=dict,
        description="Node failure-mode vocabulary: value -> registry entry",
    )
    edge_mechanisms: dict[str, VocabEntry] = Field(
        default_factory=dict,
        description="Edge propagation-mechanism vocabulary: value -> registry entry",
    )
    entity_types: dict[str, EntityTypeSpec] = Field(
        default_factory=dict,
        description="The system's entity-type set, declared in its profile "
        "just like the vocabularies",
    )

    @field_validator("node_predicates", "edge_mechanisms", "entity_types")
    @classmethod
    def _check_value_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        import re

        for key in v:
            if not re.match(_VALUE_PATTERN, key):
                raise ValueError(
                    f"vocabulary value {key!r} must match {_VALUE_PATTERN}"
                )
            if key in RESERVED_VALUES:
                raise ValueError(
                    f"vocabulary value {key!r} is reserved (factory-injected)"
                )
        return v

    @property
    def vocab_version(self) -> str:
        """The vocab_version string scenario files declare, e.g.
        'core-0.1.0+ecom-0.1.0'.
        """
        return "+".join([*self.lineage, f"{self.name}-{self.version}"])

    def extend(
        self,
        name: str,
        version: str,
        *,
        node_predicates: dict[str, VocabEntry] | None = None,
        edge_mechanisms: dict[str, VocabEntry] | None = None,
        entity_types: dict[str, EntityTypeSpec] | None = None,
    ) -> "VocabProfile":
        """Compose a new profile on top of this one. Duplicate values are an
        error — extensions add, they never silently override.
        """

        def merged(base: dict, extra: dict | None, what: str) -> dict:
            extra = extra or {}
            dup = sorted(set(base) & set(extra))
            if dup:
                raise ValueError(f"extension {name!r} redefines existing {what}: {dup}")
            return {**base, **extra}

        return VocabProfile(
            name=name,
            version=version,
            lineage=[*self.lineage, f"{self.name}-{self.version}"],
            node_predicates=merged(
                self.node_predicates, node_predicates, "node_predicates"
            ),
            edge_mechanisms=merged(
                self.edge_mechanisms, edge_mechanisms, "edge_mechanisms"
            ),
            entity_types=merged(self.entity_types, entity_types, "entity_types"),
        )

    def entity_type_objects(self) -> list[EntityType]:
        """Materialize this profile's entity types for registry construction."""
        return [
            EntityType(
                prefix=prefix,
                brief=spec.brief,
                stability=spec.stability,
                since=spec.since,
                parent=spec.parent,
                namespace=self.name,
            )
            for prefix, spec in self.entity_types.items()
        ]


def _read_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        return tomllib.loads(text)
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    raise ValueError(f"unsupported profile format {suffix!r} (use .json/.toml/.yaml)")


# The core profile ships as package data (fpg/profiles/core.toml) and is
# loaded once at import. It is intentionally EMPTY — no vocabulary; it only
# anchors the version lineage that system profiles extend.
def _load_packaged_core() -> VocabProfile:
    data = tomllib.loads(
        resources.files("fpg").joinpath("profiles/core.toml").read_text()
    )
    data.pop("extends", None)
    return VocabProfile(**data)


CORE_PROFILE = _load_packaged_core()


def load_profile(path: str | Path, *, base: VocabProfile | None = None) -> VocabProfile:
    """Load a profile from a config file.

    ``extends = "core"`` (the default) composes the file's entries on top of
    ``base`` (CORE_PROFILE when omitted); ``extends = "none"`` makes the
    file self-contained (it must then bring its full vocabulary).
    """
    base = base if base is not None else CORE_PROFILE
    path = Path(path)
    data = _read_file(path)
    extends = data.pop("extends", "core")
    if extends == "none":
        return VocabProfile(**data)
    if extends != base.name:
        raise ValueError(
            f"profile {path.name!r} extends {extends!r}, "
            f"but base profile is {base.name!r}"
        )
    return base.extend(
        data["name"],
        data["version"],
        node_predicates={
            k: VocabEntry(**v) for k, v in data.get("node_predicates", {}).items()
        },
        edge_mechanisms={
            k: VocabEntry(**v) for k, v in data.get("edge_mechanisms", {}).items()
        },
        entity_types={
            k: EntityTypeSpec(**v) for k, v in data.get("entity_types", {}).items()
        },
    )
