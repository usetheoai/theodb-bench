"""Loading and validating the versioned machine-readable artifacts.

Every artifact this project writes carries a ``schema_version`` and is validated
against a schema in ``schemas/`` before it is written and after it is read. A
run bundle that cannot be validated is not evidence, so validation failures are
loud and name the exact JSON path that broke.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Final

import jsonschema
from theodb_bench.errors import ErrorContext, Phase, SchemaValidationError

SCHEMA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "schemas"

SCHEMA_NAMES: Final[tuple[str, ...]] = (
    "benchmark",
    "manifest",
    "environment",
    "dataset",
    "system",
    "validation",
    "result",
    "statistics",
    "regression",
    "pareto",
    "summary",
)


def schema_path(name: str) -> Path:
    """Resolve a schema by name, refusing anything that escapes the schema dir."""
    if name not in SCHEMA_NAMES:
        raise SchemaValidationError(
            f"unknown schema {name!r}; known schemas: {', '.join(SCHEMA_NAMES)}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    return SCHEMA_DIR / f"{name}.schema.json"


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by name. Cached: schemas are immutable at runtime."""
    path = schema_path(name)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaValidationError(
            f"schema {name!r} could not be read from {path}",
            context=ErrorContext(phase=Phase.OFFLINE),
            cause=exc,
        ) from exc
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    schema = load_schema(name)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    built: jsonschema.protocols.Validator = validator_cls(schema)
    return built


def _json_pointer(error: jsonschema.ValidationError) -> str:
    if not error.absolute_path:
        return "$"
    return "$." + ".".join(str(part) for part in error.absolute_path)


def validate(name: str, payload: Any, *, context: ErrorContext | None = None) -> None:
    """Validate a payload against a named schema.

    Raises ``SchemaValidationError`` listing every violation rather than only
    the first, so a malformed artifact is fixed in one pass.
    """
    errors = sorted(_validator(name).iter_errors(payload), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    rendered = "; ".join(f"{_json_pointer(err)}: {err.message}" for err in errors)
    ctx = context if context is not None else ErrorContext(phase=Phase.OFFLINE)
    ctx.details.setdefault("schema", name)
    ctx.details.setdefault("violations", [_json_pointer(err) for err in errors])
    raise SchemaValidationError(f"{name} artifact failed validation: {rendered}", context=ctx)


def read_validated(name: str, path: Path, *, context: ErrorContext | None = None) -> Any:
    """Read a JSON artifact from disk and validate it before returning it."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaValidationError(
            f"{name} artifact could not be read from {path}",
            context=context if context is not None else ErrorContext(phase=Phase.OFFLINE),
            cause=exc,
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"{name} artifact at {path} is not valid JSON: {exc}",
            context=context if context is not None else ErrorContext(phase=Phase.OFFLINE),
            cause=exc,
        ) from exc
    validate(name, payload, context=context)
    return payload


def write_validated(
    name: str, path: Path, payload: Any, *, context: ErrorContext | None = None
) -> None:
    """Validate a payload and only then write it.

    Writing first and validating later would leave an invalid artifact in a
    bundle that is supposed to be immutable.
    """
    validate(name, payload, context=context)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
