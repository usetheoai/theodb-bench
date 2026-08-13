"""A regression gate that cannot tell noise from a change must say so."""

from __future__ import annotations

import pytest
from theodb_bench.analysis.regression import (
    ADVISORY_SOURCE,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MEASURED,
    BaselineRef,
    Candidate,
    Gate,
    check_comparability,
    compare,
    evaluate_gate,
    gates_from_noise_floor,
)
from theodb_bench.schemas import validate


def _baseline(**overrides: object) -> BaselineRef:
    base: dict[str, object] = {
        "run_id": "20260801T120000Z-vector-sift1m-theodb-aaa111",
        "benchmark_id": "vector/sift1m/hnsw",
        "benchmark_version": 1,
        "profile": "pr",
        "system": "theodb",
        "hardware_class": "bench-01",
        "metrics": {"throughput_per_second": 400.0, "recall": 0.98, "latency_p99_ms": 5.0},
    }
    base.update(overrides)
    return BaselineRef(**base)  # type: ignore[arg-type]


def _candidate(**overrides: object) -> Candidate:
    base: dict[str, object] = {
        "run_id": "20260813T120000Z-vector-sift1m-theodb-bbb222",
        "benchmark_id": "vector/sift1m/hnsw",
        "benchmark_version": 1,
        "profile": "pr",
        "system": "theodb",
        "hardware_class": "bench-01",
        "metrics": {"throughput_per_second": 400.0, "recall": 0.98, "latency_p99_ms": 5.0},
    }
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


_MEASURED_GATE = Gate(
    metric="throughput_per_second",
    direction=HIGHER_IS_BETTER,
    max_regression_pct=3.0,
    threshold_source=MEASURED,
)


# ------------------------------------------------------------- comparability


def test_identical_configuration_is_comparable() -> None:
    comparable, checks = check_comparability(_candidate(), _baseline())
    assert comparable
    assert all(check["matches"] for check in checks)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark_version", 2),
        ("hardware_class", "laptop"),
        ("profile", "nightly"),
        ("system", "pgvector"),
        ("benchmark_id", "vector/other/hnsw"),
    ],
)
def test_any_configuration_difference_breaks_comparability(field: str, value: object) -> None:
    comparable, _ = check_comparability(_candidate(**{field: value}), _baseline())
    assert not comparable


def test_a_missing_baseline_is_not_comparable() -> None:
    comparable, checks = check_comparability(_candidate(), None)
    assert not comparable
    assert checks[0]["field"] == "baseline"


def test_an_incomparable_baseline_yields_incomparable_not_pass() -> None:
    # The whole point: silently comparing across hardware lets a real
    # regression be explained away.
    payload = compare(_candidate(hardware_class="laptop"), _baseline(), [_MEASURED_GATE])
    assert payload["verdict"] == "INCOMPARABLE"
    assert payload["gates"] == []
    assert "explained away" in payload["notes"]


def test_no_baseline_at_all_yields_incomparable() -> None:
    assert compare(_candidate(), None, [_MEASURED_GATE])["verdict"] == "INCOMPARABLE"


# --------------------------------------------------------------------- gates


def test_an_unchanged_metric_passes() -> None:
    outcome = evaluate_gate(_MEASURED_GATE, _candidate(), _baseline())
    assert outcome.outcome == "PASS"
    assert outcome.delta_pct == pytest.approx(0.0)


def test_an_improvement_passes_and_reports_a_negative_regression() -> None:
    candidate = _candidate(metrics={"throughput_per_second": 500.0})
    outcome = evaluate_gate(_MEASURED_GATE, candidate, _baseline())
    assert outcome.outcome == "PASS"
    assert outcome.delta_pct is not None and outcome.delta_pct < 0


def test_a_regression_beyond_a_measured_budget_fails() -> None:
    candidate = _candidate(metrics={"throughput_per_second": 300.0})
    outcome = evaluate_gate(_MEASURED_GATE, candidate, _baseline())
    assert outcome.outcome == "FAIL"
    assert "25.00% worse" in outcome.detail


def test_a_regression_within_budget_passes() -> None:
    candidate = _candidate(metrics={"throughput_per_second": 392.0})  # 2% worse
    assert evaluate_gate(_MEASURED_GATE, candidate, _baseline()).outcome == "PASS"


def test_direction_is_respected_for_latency() -> None:
    gate = Gate(
        "latency_p99_ms", LOWER_IS_BETTER, max_regression_pct=5.0, threshold_source=MEASURED
    )
    worse = evaluate_gate(gate, _candidate(metrics={"latency_p99_ms": 8.0}), _baseline())
    better = evaluate_gate(gate, _candidate(metrics={"latency_p99_ms": 4.0}), _baseline())
    assert worse.outcome == "FAIL"
    assert better.outcome == "PASS"


def test_an_absolute_budget_catches_a_small_quality_drop() -> None:
    # 0.98 -> 0.978 is 0.2% relative, which a percentage budget would wave
    # through; for recall the absolute change is what matters.
    gate = Gate(
        "recall",
        HIGHER_IS_BETTER,
        max_absolute_regression=0.001,
        threshold_source=MEASURED,
    )
    outcome = evaluate_gate(gate, _candidate(metrics={"recall": 0.978}), _baseline())
    assert outcome.outcome == "FAIL"


def test_a_metric_missing_from_either_side_is_unavailable_not_pass() -> None:
    outcome = evaluate_gate(_MEASURED_GATE, _candidate(metrics={}), _baseline())
    assert outcome.outcome == "UNAVAILABLE"
    assert "cannot show the absence of a regression" in outcome.detail


def test_a_zero_baseline_does_not_produce_a_division_error() -> None:
    outcome = evaluate_gate(
        _MEASURED_GATE,
        _candidate(metrics={"throughput_per_second": 5.0}),
        _baseline(metrics={"throughput_per_second": 0.0}),
    )
    assert outcome.delta_pct is None


# -------------------------------------------------------- threshold provenance


def test_a_guessed_threshold_only_advises() -> None:
    # Until the noise floor is measured, a breach cannot be distinguished from
    # the machine having a bad afternoon.
    guessed = Gate("throughput_per_second", HIGHER_IS_BETTER, max_regression_pct=3.0)
    outcome = evaluate_gate(
        guessed, _candidate(metrics={"throughput_per_second": 100.0}), _baseline()
    )
    assert outcome.outcome == "ADVISORY"
    assert "not derived from a measured noise floor" in outcome.detail
    assert guessed.threshold_source == ADVISORY_SOURCE


def test_gates_derived_from_the_noise_floor_are_marked_measured() -> None:
    gates = gates_from_noise_floor(
        {"throughput_per_second": 0.01}, {"throughput_per_second": HIGHER_IS_BETTER}
    )
    assert gates[0].threshold_source == MEASURED
    assert gates[0].max_regression_pct == pytest.approx(3.0)


def test_a_noisier_metric_gets_a_looser_budget() -> None:
    gates = {
        gate.metric: gate for gate in gates_from_noise_floor({"steady": 0.005, "jittery": 0.05}, {})
    }
    jittery = gates["jittery"].max_regression_pct
    steady = gates["steady"].max_regression_pct
    assert jittery is not None and steady is not None
    assert jittery > steady


def test_an_invalid_multiplier_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        gates_from_noise_floor({"qps": 0.01}, {}, multiplier=0)


def test_an_unknown_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown direction"):
        Gate("qps", "sideways")


# ------------------------------------------------------------------- verdicts


def test_a_clean_comparison_validates_and_passes() -> None:
    payload = compare(_candidate(), _baseline(), [_MEASURED_GATE])
    validate("regression", payload)
    assert payload["verdict"] == "PASS"


def test_one_failing_gate_fails_the_comparison() -> None:
    payload = compare(
        _candidate(metrics={"throughput_per_second": 100.0}), _baseline(), [_MEASURED_GATE]
    )
    validate("regression", payload)
    assert payload["verdict"] == "FAIL"


def test_an_unevaluable_gate_downgrades_the_verdict_from_pass() -> None:
    payload = compare(_candidate(metrics={}), _baseline(), [_MEASURED_GATE])
    assert payload["verdict"] == "ADVISORY"
    assert "could not be evaluated" in payload["notes"]


def test_the_noise_floor_is_recorded_alongside_the_verdict() -> None:
    payload = compare(
        _candidate(),
        _baseline(),
        [_MEASURED_GATE],
        noise_floor={"throughput_per_second": 0.01},
        noise_floor_runs=12,
    )
    validate("regression", payload)
    assert payload["noise_floor"]["source_runs"] == 12
