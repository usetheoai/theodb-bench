"""Comparing two systems, paired by query, or refusing to compare.

Invariant I14 says a comparison of two systems requires a paired test — not a
comparison of means. Measured 2026-08-17, `analysis/significance.py` implemented
that test (paired randomisation, paired bootstrap CI, Cohen's dz, Monte-Carlo
correction) and had **zero importers** in `src/`, while `render_comparison` put
medians side by side. The machinery existed and the only command that compares
systems did not call it.

The missing link was upstream of both: `VectorBenchmark.measure` collected a
latency per query and kept only the summary, so no bundle ever carried the paired
samples the test needs.

One detail decides whether the pairing is correct at all. The latency list skips
queries that errored or timed out, so position *i* in the list is not query *i*.
Pairing by position would silently misalign every sample after the first timeout
and produce a confident, wrong p-value — the exact shape of defect this repository
spent the day removing. Samples therefore carry their query id.
"""

from __future__ import annotations

import pytest
from theodb_bench.analysis.significance import compare_systems
from theodb_bench.compare import (
    PairedSamples,
    match_by_recall,
    pair_by_query,
    render_paired_verdict,
)
from theodb_bench.errors import ConfigError


def test_samples_pair_by_query_id_not_by_position() -> None:
    """The case a positional pairing gets wrong: one side lost a query."""
    a = {0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0}
    b = {0: 20.0, 1: 21.0, 3: 23.0}  # query 2 timed out on system B

    with pytest.raises(ConfigError, match="query sets differ"):
        pair_by_query(a, b)


def test_identical_query_sets_pair_in_a_stable_order() -> None:
    a = {2: 12.0, 0: 10.0, 1: 11.0}
    b = {1: 21.0, 2: 22.0, 0: 20.0}

    paired = pair_by_query(a, b)

    assert isinstance(paired, PairedSamples)
    assert paired.query_ids == (0, 1, 2)
    assert paired.a == (10.0, 11.0, 12.0)
    assert paired.b == (20.0, 21.0, 22.0)


def test_an_empty_side_is_refused_rather_than_compared() -> None:
    with pytest.raises(ConfigError, match="no paired samples"):
        pair_by_query({}, {})


def test_the_paired_result_feeds_the_significance_test() -> None:
    """End to end: the pairing produces exactly what compare_systems accepts.

    The per-query differences vary deliberately. A constant difference has zero
    standard deviation, so Cohen's dz is undefined and the implementation
    correctly returns None — correct, and not what this test is about.
    """
    a = {i: 10.0 + (i % 7) * 0.3 for i in range(40)}
    b = {i: 12.0 + (i % 5) * 0.4 for i in range(40)}

    paired = pair_by_query(a, b)
    result = compare_systems(list(paired.a), list(paired.b))

    assert result.significant is True
    assert result.effect.cohens_dz is not None


def test_two_indistinguishable_systems_are_reported_as_indistinguishable() -> None:
    """The honest negative the test exists to produce.

    A comparison of medians would still print two different numbers here and let a
    reader infer a winner. The paired test is what says there is none.
    """
    a = {i: 10.0 + (i % 3) * 0.01 for i in range(60)}
    b = {i: 10.0 + ((i + 1) % 3) * 0.01 for i in range(60)}

    result = compare_systems(list(pair_by_query(a, b).a), list(pair_by_query(a, b).b))

    assert result.significant is False


# --------------------------------------- the comparison the CLI actually renders


def test_the_comparison_reports_a_paired_verdict_not_two_medians() -> None:
    """Invariant I14, made executable.

    `render_comparison` used to put each system's median throughput and recall in
    adjacent columns and leave the reader to infer a winner. Two medians are not a
    comparison of two systems; they are two summaries printed near each other.
    """
    a = {i: 10.0 + (i % 7) * 0.3 for i in range(60)}
    b = {i: 13.0 + (i % 5) * 0.4 for i in range(60)}

    verdict = render_paired_verdict("theodb", a, "alloydbomni", b, metric="latency_ms")

    assert "p =" in verdict
    assert "95% CI" in verdict
    assert "n = 60" in verdict
    # The direction has to be named, not left to the reader.
    assert "theodb" in verdict and "alloydbomni" in verdict


def test_an_unpairable_comparison_says_so_instead_of_falling_back_to_medians() -> None:
    """Falling back would be worse than refusing: the reader sees a comparison."""
    a = dict.fromkeys(range(50), 10.0)
    b = dict.fromkeys(range(40), 11.0)

    verdict = render_paired_verdict("theodb", a, "alloydbomni", b, metric="latency_ms")

    assert "not comparable" in verdict
    assert "p =" not in verdict


def test_indistinguishable_systems_are_named_as_such() -> None:
    a = {i: 10.0 + (i % 3) * 0.01 for i in range(60)}
    b = {i: 10.0 + ((i + 1) % 3) * 0.01 for i in range(60)}

    verdict = render_paired_verdict("a", a, "b", b, metric="latency_ms")

    assert "indistinguishable" in verdict


# ------------------------------------------- pairing two engines at matched recall
#
# Measured 2026-08-17: pairing by configuration label can never compare two
# engines, because the label carries engine-specific parameters --
# `ivfflat ... probes=20` on one side and `scann ... num_leaves_to_search=20` on
# the other. The label is the right key for a regression (same system, two runs)
# and the wrong one for a comparison.
#
# The question the project asks is at MATCHED RECALL: each engine's frontier, read
# at equal quality. Two knobs named differently and set to the same integer are not
# the same operating point, and pairing them would compare the knobs.


def test_two_engines_pair_at_the_closest_matching_recall() -> None:
    a = {"probes=5": 0.7656, "probes=20": 0.9570, "probes=80": 0.9866}
    b = {"leaves=5": 0.7060, "leaves=20": 0.9594, "leaves=80": 0.9960}

    match = match_by_recall(a, b, tolerance=0.01)

    assert match is not None
    assert match.label_a == "probes=20"
    assert match.label_b == "leaves=20"
    assert abs(match.recall_a - match.recall_b) < 0.01


def test_frontiers_that_never_meet_are_refused_not_forced() -> None:
    """The AH-without-rescore case: one frontier tops out below the other's floor."""
    a = {"probes=20": 0.9570, "probes=80": 0.9866}
    b = {"leaves=20": 0.6440, "leaves=80": 0.6582}

    assert match_by_recall(a, b, tolerance=0.01) is None


def test_the_tolerance_is_honoured_rather_than_taking_the_nearest_at_any_distance() -> None:
    a = {"probes=20": 0.9570}
    b = {"leaves=20": 0.9000}

    assert match_by_recall(a, b, tolerance=0.01) is None
    assert match_by_recall(a, b, tolerance=0.10) is not None
