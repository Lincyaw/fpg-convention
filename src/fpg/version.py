"""Versioning policy, modeled after OpenTelemetry semantic conventions.

Two independently versioned artifact families, both using semver:

``SCHEMA_VERSION`` (here) — the *structural* schema (field layout of
Scenario / ModelRCAOutput and their validation rules).
  - MAJOR: breaking change (field removed/renamed, constraint tightened so
    that previously valid files become invalid);
  - MINOR: additive change (new optional field, new validation that only
    rejects previously undefined constructs);
  - PATCH: documentation/description clarification only.

Vocabulary versions (NOT here) — each profile carries its own ``version``
field; the core vocabulary's version lives in fpg/profiles/core.toml and is
exposed as ``fpg.CORE_PROFILE.version``.
  - MAJOR: value removed (only allowed after a deprecation cycle);
  - MINOR: value added, or stability promoted (experimental -> stable);
  - PATCH: brief/description clarification only.

Evolution rules (semconv-style):
  - New vocabulary entries start as ``experimental`` and are promoted to
    ``stable`` once attested in real annotated scenarios.
  - Renames never delete: the old value is marked ``deprecated`` with
    ``renamed_to`` set, kept for at least one MINOR release, then removed
    in the next MAJOR release.
  - Ground-truth scenario files are self-describing: they carry
    ``schema_version`` and ``vocab_version`` so that consumers can migrate.
  - Testbed-specific extension vocabularies version independently; a
    scenario declares the combination it uses, e.g.
    ``core-0.1.0+ecom-0.1.0`` (see Scenario.vocab_version).
"""

SCHEMA_VERSION = "0.1.0"
