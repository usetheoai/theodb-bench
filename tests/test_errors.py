"""Errors must carry enough context to diagnose a run without a debugger."""

from __future__ import annotations

import json

from theodb_bench.errors import (
    BenchError,
    ChecksumMismatchError,
    ErrorContext,
    Phase,
    Recoverability,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)


def test_error_serialises_full_context() -> None:
    err = ChecksumMismatchError(
        "sift1m/base.fvecs: expected sha256 abc, got def",
        context=ErrorContext(
            phase=Phase.DATASET_LOAD,
            system="theodb",
            benchmark="vector/sift1m/hnsw",
            run_id="2026-08-12T231000Z-vector-sift1m-theodb-abc123",
            details={"file": "base.fvecs"},
        ),
    )
    payload = json.loads(json.dumps(err.as_dict()))
    assert payload["error"] == "ChecksumMismatchError"
    assert payload["phase"] == "dataset_load"
    assert payload["system"] == "theodb"
    assert payload["benchmark"] == "vector/sift1m/hnsw"
    assert payload["run_id"].endswith("abc123")
    assert payload["details"] == {"file": "base.fvecs"}
    assert payload["recoverability"] == "fatal"


def test_transport_failure_is_recoverable_but_protocol_failure_is_not() -> None:
    # Retrying a timeout can succeed; retrying a checksum mismatch cannot,
    # and retrying it would be a way to eventually accept the wrong bytes.
    assert SystemUnavailableError("not ready").recoverability is Recoverability.RECOVERABLE
    assert ChecksumMismatchError("bad sha").recoverability is Recoverability.FATAL


def test_unsupported_capability_is_an_adapter_error_not_a_crash() -> None:
    err = UnsupportedCapabilityError("cosine opclass absent for theodb_hnsw")
    assert isinstance(err, BenchError)
    assert err.as_dict()["error"] == "UnsupportedCapabilityError"


def test_str_renders_context_but_stays_readable() -> None:
    err = BenchError("boom", context=ErrorContext(phase=Phase.MEASUREMENT, system="pgvector"))
    assert str(err) == "boom [phase=measurement system=pgvector]"


def test_str_without_context_is_just_the_message() -> None:
    assert str(BenchError("boom")) == "boom"


def test_cause_is_recorded_when_chaining() -> None:
    root = ValueError("connection refused")
    err = SystemUnavailableError("theodb did not become ready", cause=root)
    assert err.as_dict()["cause"] == "ValueError: connection refused"


def test_every_lifecycle_phase_has_a_stable_wire_value() -> None:
    # These strings land in result bundles; renaming one breaks stored runs.
    assert [p.value for p in Phase] == [
        "preflight",
        "environment",
        "isolation",
        "bootstrap",
        "dataset_load",
        "index_build",
        "warmup",
        "measurement",
        "cooldown",
        "validation",
        "finalization",
        "offline",
    ]
