"""RRF fuses by rank because scores from different legs share no unit."""

from __future__ import annotations

import pytest
from theodb_bench.analysis.fusion import (
    DEFAULT_K,
    fuse_to_ids,
    rank_agreement,
    reciprocal_rank_fusion,
)


def test_a_single_leg_is_returned_in_its_own_order() -> None:
    fused = fuse_to_ids({"vector": [7, 3, 9]}, n=3)
    assert fused == [7, 3, 9]


def test_a_document_ranked_well_by_both_legs_wins() -> None:
    fused = fuse_to_ids({"lexical": [1, 2, 3], "vector": [1, 4, 5]}, n=3)
    assert fused[0] == 1


def test_agreement_between_legs_beats_a_single_first_place() -> None:
    # 2 is second in both legs; 1 and 9 are first in one leg each. Two second
    # places outweigh one first place, which is the behaviour RRF is chosen for.
    fused = fuse_to_ids({"lexical": [1, 2], "vector": [9, 2]}, n=3)
    assert fused[0] == 2


def test_ties_are_broken_by_document_id() -> None:
    # Without this the order depends on dict iteration and the fusion stops
    # being reproducible.
    fused = fuse_to_ids({"a": [5], "b": [3]}, n=2)
    assert fused == [3, 5]


def test_weights_shift_the_balance() -> None:
    unweighted = fuse_to_ids({"lexical": [1], "vector": [2]}, n=2)
    weighted = fuse_to_ids({"lexical": [1], "vector": [2]}, n=2, weights={"vector": 10.0})
    assert unweighted[0] == 1
    assert weighted[0] == 2


def test_a_zero_weighted_leg_contributes_nothing() -> None:
    fused = fuse_to_ids({"lexical": [1, 2, 3], "vector": [9]}, n=5, weights={"vector": 0.0})
    assert 9 not in fused


def test_a_duplicate_within_a_leg_is_counted_once() -> None:
    # A leg returning the same document twice must not outvote a correct leg.
    doubled = reciprocal_rank_fusion({"buggy": [1, 1, 1], "good": [2]})
    scores = dict(doubled)
    assert scores[1] == pytest.approx(1.0 / (DEFAULT_K + 1))
    assert scores[2] == pytest.approx(1.0 / (DEFAULT_K + 1))


def test_scores_follow_the_published_formula() -> None:
    fused = dict(reciprocal_rank_fusion({"leg": [10, 20]}, k=60))
    assert fused[10] == pytest.approx(1 / 61)
    assert fused[20] == pytest.approx(1 / 62)


def test_the_constant_matters_and_is_explicit() -> None:
    tight = dict(reciprocal_rank_fusion({"leg": [1, 2]}, k=1))
    loose = dict(reciprocal_rank_fusion({"leg": [1, 2]}, k=1000))
    # A small k separates ranks sharply; a large one flattens them.
    assert (tight[1] - tight[2]) > (loose[1] - loose[2])


def test_a_non_positive_constant_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        reciprocal_rank_fusion({"leg": [1]}, k=0)


def test_empty_input_fuses_to_nothing() -> None:
    assert fuse_to_ids({}, n=5) == []
    assert fuse_to_ids({"leg": []}, n=5) == []


def test_truncation_respects_n() -> None:
    assert len(fuse_to_ids({"leg": list(range(100))}, n=7)) == 7


# ----------------------------------------------------------------- agreement


def test_identical_rankings_agree_completely() -> None:
    assert rank_agreement([1, 2, 3], [1, 2, 3], n=3) == pytest.approx(1.0)


def test_disjoint_rankings_do_not_agree() -> None:
    assert rank_agreement([1, 2], [8, 9], n=2) == pytest.approx(0.0)


def test_agreement_ignores_order_within_the_depth() -> None:
    # Overlap, not rank correlation: the two contain the same documents.
    assert rank_agreement([1, 2, 3], [3, 2, 1], n=3) == pytest.approx(1.0)


def test_partial_overlap_is_proportional() -> None:
    assert rank_agreement([1, 2, 3, 4], [1, 2, 9, 8], n=4) == pytest.approx(0.5)


def test_two_empty_rankings_agree_vacuously() -> None:
    assert rank_agreement([], [], n=5) == pytest.approx(1.0)


def test_a_non_positive_depth_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        rank_agreement([1], [1], n=0)
