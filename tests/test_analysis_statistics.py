"""Aggregation must expose the spread, not replace it with a single number."""

from __future__ import annotations

import numpy as np
import pytest
from theodb_bench.absent import Absent
from theodb_bench.analysis.statistics import (
    aggregate,
    assess_stability,
    noise_floor,
    statistics_payload,
    summarise_latency,
    summarise_points,
    throughput_best_of_n,
)
from theodb_bench.errors import ConfigError
from theodb_bench.schemas import validate

# -------------------------------------------------------------------- latency


def test_percentiles_match_numpy_on_a_known_sample() -> None:
    samples = [float(i) for i in range(1, 1001)]
    summary = summarise_latency(samples)
    assert summary.p50 == pytest.approx(float(np.percentile(samples, 50)))
    assert summary.p95 == pytest.approx(float(np.percentile(samples, 95)))
    assert summary.p99 == pytest.approx(float(np.percentile(samples, 99)))


def test_p999_is_withheld_when_the_sample_is_too_small() -> None:
    # With 50 samples, p99.9 is the maximum wearing a percentile's name.
    summary = summarise_latency([float(i) for i in range(50)])
    assert isinstance(summary.p999, Absent)
    assert "1000 samples" in str(summary.p999)


def test_p999_is_reported_once_the_sample_supports_it() -> None:
    summary = summarise_latency([float(i) for i in range(2000)])
    assert not isinstance(summary.p999, Absent)


def test_no_successful_operations_produces_absences_not_zeros() -> None:
    summary = summarise_latency([])
    assert summary.sample_count == 0
    assert isinstance(summary.p50, Absent)
    assert isinstance(summary.mean, Absent)


def test_a_single_sample_has_no_standard_deviation() -> None:
    summary = summarise_latency([4.2])
    assert summary.mean == pytest.approx(4.2)
    assert isinstance(summary.stdev, Absent)


def test_latency_summary_serialises_absences_explicitly() -> None:
    payload = summarise_latency([]).as_dict()
    assert payload["p50"] == {
        "absent": "unavailable",
        "detail": "no successful operations were recorded",
    }


# ----------------------------------------------------------------- throughput


def test_throughput_uses_the_fastest_round() -> None:
    # ANN-Benchmarks best-of-N: 0.002 s/op is the fastest round, so 500 ops/s.
    assert throughput_best_of_n([0.004, 0.002, 0.003]) == pytest.approx(500.0)


def test_throughput_without_a_usable_round_is_absent() -> None:
    assert isinstance(throughput_best_of_n([0.0, -1.0]), Absent)


# ------------------------------------------------------------------ aggregate


def test_aggregate_keeps_every_repetition() -> None:
    result = aggregate([10.0, 12.0, 11.0])
    assert result.values == (10.0, 12.0, 11.0)
    assert result.repetitions == 3
    assert result.median == pytest.approx(11.0)


def test_aggregate_reports_spread() -> None:
    result = aggregate([10.0, 20.0])
    assert result.minimum == 10.0
    assert result.maximum == 20.0
    assert not isinstance(result.stdev, Absent)


def test_a_single_repetition_has_no_interval_and_says_so() -> None:
    result = aggregate([7.0])
    assert isinstance(result.stdev, Absent)
    assert isinstance(result.ci95_low, Absent)
    assert isinstance(result.coefficient_of_variation, Absent)


def test_zero_mean_makes_the_coefficient_of_variation_undefined() -> None:
    result = aggregate([-1.0, 1.0])
    assert isinstance(result.coefficient_of_variation, Absent)


def test_aggregating_nothing_is_refused() -> None:
    with pytest.raises(ConfigError, match="zero repetitions"):
        aggregate([])


def test_confidence_interval_brackets_the_mean() -> None:
    result = aggregate([100.0, 102.0, 98.0, 101.0, 99.0])
    assert not isinstance(result.ci95_low, Absent)
    assert not isinstance(result.ci95_high, Absent)
    assert result.ci95_low < result.mean < result.ci95_high


# ------------------------------------------------------------------ stability


def test_tight_repetitions_are_stable() -> None:
    assert assess_stability({"qps": aggregate([100.0, 101.0, 99.5])}).stable


def test_noisy_repetitions_are_flagged_but_kept() -> None:
    # Instability is reported, never corrected: the point stays in the result.
    stability = assess_stability({"qps": aggregate([100.0, 400.0, 50.0])})
    assert not stability.stable
    assert "qps cv=" in stability.reason


def test_stability_without_any_spread_is_not_claimed() -> None:
    assert not assess_stability({"qps": aggregate([100.0])}).stable


# --------------------------------------------------------------------- points


def test_summarise_points_produces_a_valid_statistics_artifact() -> None:
    points = summarise_points(
        [
            (
                "hnsw ef=64",
                {"ef_search": 64},
                {
                    "throughput_per_second": [400.0, 398.0, 402.0],
                    "recall_at_10": [0.98, 0.98, 0.99],
                },
            )
        ]
    )
    payload = statistics_payload("20260813T000000Z-vector-fake-abcdef", points)
    validate("statistics", payload)
    assert payload["outlier_policy"] == {"name": "none"}


def test_default_outlier_policy_removes_nothing() -> None:
    points = summarise_points([("p", {}, {"qps": [1.0, 1000.0]})])
    payload = statistics_payload("run", points)
    # The wild value is still there; nothing was dropped behind the reader's back.
    assert payload["points"][0]["metrics"]["qps"]["values"] == [1.0, 1000.0]


# ---------------------------------------------------------------- noise floor


def test_noise_floor_measures_variation_across_identical_runs() -> None:
    floor = noise_floor([{"qps": 100.0}, {"qps": 102.0}, {"qps": 98.0}])
    assert 0.0 < floor["qps"] < 0.1


def test_noise_floor_needs_more_than_one_run() -> None:
    with pytest.raises(ConfigError, match="at least 2 runs"):
        noise_floor([{"qps": 100.0}])
