"""Schemas are the contract for every artifact a run leaves behind.

Positive and negative cases both matter: a schema that accepts everything
documents nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from theodb_bench.errors import SchemaValidationError
from theodb_bench.schemas import (
    SCHEMA_NAMES,
    load_schema,
    read_validated,
    schema_path,
    validate,
    write_validated,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_is_itself_a_valid_schema(name: str) -> None:
    schema = load_schema(name)
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert schema["title"]


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_requires_a_schema_version(name: str) -> None:
    # Without this, an artifact written by a future version is indistinguishable
    # from one written today, and stored runs stop being readable.
    schema = load_schema(name)
    assert "schema_version" in schema["required"]
    assert schema["properties"]["schema_version"]["const"] == 1


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_valid_fixture_passes(name: str) -> None:
    validate(name, _fixture(name))


def test_unknown_schema_name_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="unknown schema"):
        schema_path("../../etc/passwd")


def test_missing_schema_version_fails() -> None:
    payload = _fixture("manifest")
    del payload["schema_version"]
    with pytest.raises(SchemaValidationError, match="schema_version"):
        validate("manifest", payload)


def test_wrong_schema_version_fails() -> None:
    payload = _fixture("manifest")
    payload["schema_version"] = 2
    with pytest.raises(SchemaValidationError):
        validate("manifest", payload)


def test_unknown_status_is_rejected() -> None:
    payload = _fixture("manifest")
    payload["status"] = "PROBABLY_FINE"
    with pytest.raises(SchemaValidationError, match="status"):
        validate("manifest", payload)


def test_unknown_top_level_field_is_rejected() -> None:
    # additionalProperties: false everywhere, so a typo in a field name fails
    # loudly instead of silently dropping data out of the bundle.
    payload = _fixture("manifest")
    payload["stauts"] = "VALID"
    with pytest.raises(SchemaValidationError):
        validate("manifest", payload)


def test_malformed_sha256_is_rejected() -> None:
    payload = _fixture("dataset")
    payload["files"][0]["sha256"] = "not-a-digest"
    with pytest.raises(SchemaValidationError, match="sha256"):
        validate("dataset", payload)


def test_dataset_file_path_may_not_be_absolute() -> None:
    payload = _fixture("dataset")
    payload["files"][0]["path"] = "/etc/passwd"
    with pytest.raises(SchemaValidationError):
        validate("dataset", payload)


def test_open_loop_workload_requires_a_target_rate() -> None:
    # Without a target rate there is no coordinated-omission correction to make,
    # so an open-loop declaration without one is meaningless.
    payload = _fixture("benchmark")
    payload["workload"]["loop"] = "open"
    payload["workload"].pop("target_rate_per_second", None)
    with pytest.raises(SchemaValidationError):
        validate("benchmark", payload)


def test_result_requires_success_error_and_timeout_counts() -> None:
    payload = _fixture("result")
    del payload["points"][0]["repetitions"][0]["operations"]["timeout"]
    with pytest.raises(SchemaValidationError, match="timeout"):
        validate("result", payload)


def test_absence_record_is_accepted_where_a_number_would_be() -> None:
    payload = _fixture("result")
    payload["points"][0]["repetitions"][0]["resources"]["cycles"] = {
        "absent": "unavailable",
        "detail": "perf events not permitted",
    }
    validate("result", payload)


def test_absence_record_with_unknown_reason_is_rejected() -> None:
    payload = _fixture("result")
    payload["points"][0]["repetitions"][0]["resources"]["cycles"] = {"absent": "dunno"}
    with pytest.raises(SchemaValidationError):
        validate("result", payload)


def test_regression_verdict_enumerates_incomparable() -> None:
    payload = _fixture("regression")
    payload["verdict"] = "INCOMPARABLE"
    payload["comparability"]["comparable"] = False
    validate("regression", payload)


def test_error_message_names_every_violation_not_just_the_first() -> None:
    payload = _fixture("manifest")
    payload["status"] = "NOPE"
    payload["schema_version"] = 99
    with pytest.raises(SchemaValidationError) as excinfo:
        validate("manifest", payload)
    assert "schema_version" in str(excinfo.value)
    assert "status" in str(excinfo.value)


def test_write_validated_refuses_to_write_an_invalid_artifact(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    payload = _fixture("manifest")
    payload["status"] = "NOPE"
    with pytest.raises(SchemaValidationError):
        write_validated("manifest", target, payload)
    assert not target.exists()


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    payload = _fixture("manifest")
    write_validated("manifest", target, payload)
    assert read_validated("manifest", target) == payload


def test_read_validated_reports_unparseable_json(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="not valid JSON"):
        read_validated("manifest", target)


def test_read_validated_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="could not be read"):
        read_validated("manifest", tmp_path / "absent.json")


def test_validation_error_carries_the_schema_name_in_context() -> None:
    payload = _fixture("manifest")
    payload["status"] = "NOPE"
    with pytest.raises(SchemaValidationError) as excinfo:
        validate("manifest", payload)
    assert excinfo.value.context.details["schema"] == "manifest"
