"""Recall is a distance threshold, not id overlap. These tests pin that down."""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.analysis.quality import (
    brute_force_ground_truth,
    mrr_at_k,
    ndcg_at_k,
    neighbors_ground_truth,
    pairwise_distances,
    recall_at_k,
    recall_at_n,
    recall_from_ids,
    success_at_k,
)
from theodb_bench.errors import ConfigError


def _corpus(rows: int = 50, dimension: int = 6, seed: int = 11) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, dimension)).astype(np.float32)


# ----------------------------------------------------------------- distances


def test_l2_distance_of_a_vector_to_itself_is_zero() -> None:
    corpus = _corpus()
    distances = pairwise_distances(corpus, corpus[:3], "l2")
    for i in range(3):
        assert distances[i, i] == pytest.approx(0.0, abs=1e-9)


def test_l2_is_never_negative_despite_the_expanded_form() -> None:
    # The expansion |q|^2 + |c|^2 - 2qc can go slightly negative in floating
    # point; a negative distance would sort ahead of a true zero.
    corpus = _corpus(200, 32)
    assert pairwise_distances(corpus, corpus, "l2").min() >= 0.0


def test_cosine_of_a_zero_vector_is_maximum_distance_not_nan() -> None:
    corpus = np.zeros((2, 4), dtype=np.float32)
    distances = pairwise_distances(corpus, np.ones((1, 4), dtype=np.float32), "cosine")
    assert np.isfinite(distances).all()
    assert distances.max() == pytest.approx(1.0)


def test_dimension_mismatch_is_refused() -> None:
    with pytest.raises(ConfigError, match="dimension mismatch"):
        pairwise_distances(_corpus(10, 4), _corpus(2, 8), "l2")


def test_unknown_metric_is_refused() -> None:
    with pytest.raises(ConfigError, match="unknown metric"):
        pairwise_distances(_corpus(), _corpus(2), "manhattan")


# -------------------------------------------------------------- ground truth


def test_ground_truth_puts_each_vector_first_for_itself() -> None:
    corpus = _corpus()
    ids, distances = brute_force_ground_truth(corpus, corpus[:5], k=3)
    assert list(ids[:, 0]) == [0, 1, 2, 3, 4]
    assert distances[:, 0] == pytest.approx(np.zeros(5), abs=1e-9)


def test_ground_truth_distances_are_sorted() -> None:
    _, distances = brute_force_ground_truth(_corpus(), _corpus(4), k=5)
    assert np.all(np.diff(distances, axis=1) >= -1e-12)


def test_ground_truth_breaks_ties_by_id() -> None:
    corpus = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
    ids, _ = brute_force_ground_truth(corpus, corpus[:1], k=4)
    assert list(ids[0]) == [0, 1, 2, 3]


def test_k_beyond_the_corpus_is_refused() -> None:
    with pytest.raises(ConfigError, match="exceeds the corpus size"):
        brute_force_ground_truth(_corpus(5), _corpus(1), k=10)


def test_the_oracle_rounds_to_float32_before_computing() -> None:
    # Two values that differ only below float32 precision must be treated as
    # equal, because the system storing float4 cannot tell them apart either.
    base = np.array([[1.0]], dtype=np.float64)
    nudged = base + 1e-9
    assert pairwise_distances(base, nudged, "l2")[0, 0] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------- published neighbour lists


def test_published_neighbours_are_recomputed_not_trusted() -> None:
    corpus = _corpus()
    queries = corpus[:4]
    ids, expected = brute_force_ground_truth(corpus, queries, k=3)
    recomputed = neighbors_ground_truth(corpus, queries, ids, k=3)
    assert recomputed == pytest.approx(expected, abs=1e-9)


def test_a_negative_neighbour_id_fails_loudly() -> None:
    # NumPy would wrap this into a real vector and produce confident, wrong
    # ground truth.
    corpus = _corpus(10)
    ids = np.array([[-1, 0, 1]], dtype=np.int64)
    with pytest.raises(ConfigError, match="outside the corpus"):
        neighbors_ground_truth(corpus, corpus[:1], ids, k=3)


def test_an_out_of_range_neighbour_id_fails_loudly() -> None:
    corpus = _corpus(10)
    ids = np.array([[0, 1, 999]], dtype=np.int64)
    with pytest.raises(ConfigError, match="outside the corpus"):
        neighbors_ground_truth(corpus, corpus[:1], ids, k=3)


def test_mismatched_neighbour_row_count_is_refused() -> None:
    corpus = _corpus(10)
    with pytest.raises(ConfigError, match="neighbour rows"):
        neighbors_ground_truth(corpus, corpus[:3], np.zeros((2, 3), dtype=np.int64), k=3)


# --------------------------------------------------------------------- recall


def test_perfect_results_give_recall_one() -> None:
    corpus = _corpus()
    queries = corpus[:8]
    _, truth = brute_force_ground_truth(corpus, queries, k=5)
    assert recall_at_k(truth, truth, k=5) == pytest.approx(1.0)


def test_completely_wrong_results_give_recall_zero() -> None:
    corpus = _corpus()
    queries = corpus[:8]
    _, truth = brute_force_ground_truth(corpus, queries, k=5)
    hopeless = np.full_like(truth, 1e9)
    assert recall_at_k(truth, hopeless, k=5) == pytest.approx(0.0)


def test_half_correct_results_give_half_recall() -> None:
    truth = np.tile(np.array([[0.1, 0.2, 0.3, 0.4]]), (4, 1))
    run = np.tile(np.array([[0.1, 0.2, 9.0, 9.0]]), (4, 1))
    assert recall_at_k(truth, run, k=4) == pytest.approx(0.5)


def test_a_tied_distance_counts_as_a_hit() -> None:
    # This is exactly where id overlap and the distance threshold disagree: a
    # different vector at an identical distance is an equally good answer.
    truth = np.array([[1.0, 2.0, 3.0]])
    run = np.array([[1.0, 2.0, 3.0]])
    assert recall_at_k(truth, run, k=3) == pytest.approx(1.0)


def test_duplicated_vectors_do_not_depress_recall() -> None:
    corpus = np.vstack([_corpus(10), _corpus(10)])  # every vector appears twice
    queries = corpus[:5]
    _, truth = brute_force_ground_truth(corpus, queries, k=4)
    # A system returning the duplicates instead of the originals is correct.
    run = truth.copy()
    assert recall_at_k(truth, run, k=4) == pytest.approx(1.0)


def test_eps_admits_a_distance_just_above_the_threshold() -> None:
    truth = np.array([[1.0, 2.0]])
    run = np.array([[1.0, 2.0 + 5e-4]])
    assert recall_at_k(truth, run, k=2, eps=1e-3) == pytest.approx(1.0)
    assert recall_at_k(truth, run, k=2, eps=0.0) == pytest.approx(0.5)


def test_recall_from_ids_matches_recall_from_distances() -> None:
    corpus = _corpus()
    queries = corpus[:6]
    ids, truth = brute_force_ground_truth(corpus, queries, k=4)
    assert recall_from_ids(corpus, queries, ids, truth, k=4) == pytest.approx(1.0)


def test_recall_rejects_too_few_ground_truth_neighbours() -> None:
    with pytest.raises(ConfigError, match="ground truth has"):
        recall_at_k(np.zeros((2, 2)), np.zeros((2, 5)), k=5)


def test_recall_rejects_a_row_count_mismatch() -> None:
    with pytest.raises(ConfigError, match="rows"):
        recall_at_k(np.zeros((3, 5)), np.zeros((2, 5)), k=5)


# ------------------------------------------------------- retrieval quality


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    relevance = {1: 3.0, 2: 2.0, 3: 1.0}
    assert ndcg_at_k([1, 2, 3], relevance, k=3) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking() -> None:
    relevance = {1: 3.0, 2: 2.0, 3: 1.0}
    assert ndcg_at_k([3, 2, 1], relevance, k=3) < 1.0


def test_ndcg_without_judgements_is_zero_not_an_error() -> None:
    assert ndcg_at_k([1, 2], {}, k=2) == 0.0


def test_recall_at_n_counts_the_relevant_set() -> None:
    assert recall_at_n([1, 2, 3, 4], {2, 4, 9}, n=4) == pytest.approx(2 / 3)


def test_mrr_uses_the_first_relevant_position() -> None:
    assert mrr_at_k([9, 8, 3], {3}, k=3) == pytest.approx(1 / 3)
    assert mrr_at_k([3, 8, 9], {3}, k=3) == pytest.approx(1.0)
    assert mrr_at_k([9, 8, 7], {3}, k=3) == 0.0


def test_success_at_k_is_binary() -> None:
    assert success_at_k([5, 6], {6}, k=2) == 1.0
    assert success_at_k([5, 6], {7}, k=2) == 0.0


# ------------------------------- the oracle must fit in memory to be usable
#
# Measured on a 16 GB host: `brute_force_ground_truth` on 1M x 128 with 500
# queries was killed by the OOM killer at 10.5 GB anon RSS, for a dataset of
# 512 MB. The cause was line-level, not algorithmic scale:
#
#   np.tile(np.arange(1_000_000), (500, 1))   -> 4 GB of int64, purely to break ties
#   pairwise_distances(corpus, queries)       -> 500 x 1M float64 = 4 GB
#   np.lexsort(..., axis=1)                   -> a full sort of 1M per query for k=10
#
# Changing an oracle is the most dangerous edit possible in a benchmark harness,
# so these tests pin the exact behaviour -- including the tie-break by id -- before
# the implementation is allowed to change.


def _reference_ground_truth(
    corpus: npt.NDArray[np.floating[Any]],
    queries: npt.NDArray[np.floating[Any]],
    k: int,
    metric: str = "l2",
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """The original implementation, kept here as the equivalence oracle."""
    distances = pairwise_distances(corpus, queries, metric)
    corpus_size = distances.shape[1]
    ids = np.lexsort((np.tile(np.arange(corpus_size), (distances.shape[0], 1)), distances), axis=1)
    ids = ids[:, :k]
    ordered = np.take_along_axis(distances, ids, axis=1)
    return ids.astype(np.int64), np.asarray(ordered, dtype=np.float64)


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_ground_truth_matches_the_reference_implementation(metric: str) -> None:
    rng = np.random.default_rng(20260817)
    corpus = rng.random((400, 16), dtype=np.float32)
    queries = rng.random((7, 16), dtype=np.float32)

    ids, dists = brute_force_ground_truth(corpus, queries, 10, metric)
    ref_ids, ref_dists = _reference_ground_truth(corpus, queries, 10, metric)

    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_allclose(dists, ref_dists, rtol=1e-12, atol=0.0)


def test_ties_still_break_by_id() -> None:
    """Duplicate vectors make every distance identical, so only the id ordering
    can decide -- and it must stay ascending, or recall stops being reproducible."""
    corpus = np.ones((50, 4), dtype=np.float32)
    queries = np.zeros((3, 4), dtype=np.float32)

    ids, _ = brute_force_ground_truth(corpus, queries, 5, "l2")

    for row in ids:
        np.testing.assert_array_equal(row, np.arange(5))


def test_ties_spanning_the_top_k_boundary_are_resolved_by_id() -> None:
    """The case a partition-based shortcut gets wrong if written carelessly.

    Ten vectors sit at distance 1 and ten at distance 2; k=15 therefore cuts
    through the second group, and which five of those ten are returned must be
    the five smallest ids rather than whichever the partition happened to place.
    """
    corpus = np.zeros((20, 1), dtype=np.float32)
    corpus[:10, 0] = 1.0
    corpus[10:, 0] = 2.0
    queries = np.zeros((1, 1), dtype=np.float32)

    ids, _ = brute_force_ground_truth(corpus, queries, 15, "l2")

    np.testing.assert_array_equal(ids[0], np.arange(15))


def test_k_equal_to_the_corpus_size_is_allowed() -> None:
    rng = np.random.default_rng(1)
    corpus = rng.random((12, 3), dtype=np.float32)
    queries = rng.random((2, 3), dtype=np.float32)

    ids, _ = brute_force_ground_truth(corpus, queries, 12, "l2")

    assert ids.shape == (2, 12)


def test_the_oracle_does_not_allocate_the_full_distance_matrix() -> None:
    """A 16 GB host must be able to build ground truth for a million vectors.

    Asserted on the peak allocation rather than on wall time: the previous
    implementation's cost was two 4 GB arrays, and a corpus this size makes the
    difference between 'measurable' and 'OOM'.
    """
    rng = np.random.default_rng(2)
    corpus = rng.random((200_000, 8), dtype=np.float32)
    queries = rng.random((256, 8), dtype=np.float32)

    tracemalloc.start()
    brute_force_ground_truth(corpus, queries, 10, "l2")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The full matrix would be 256 * 200_000 * 8 = 410 MB, plus an equal tile.
    assert peak < 200_000_000, f"peak was {peak / 1e6:.0f} MB"
