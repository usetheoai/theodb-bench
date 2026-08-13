"""Retrieval quality, computed by the benchmark rather than trusted from the system.

Recall here follows ANN-Benchmarks (Aumüller, Bernhardsson & Faithfull,
arXiv:1807.05614 §2.1): a returned neighbour counts when its distance is within
``eps`` of the k-th true distance. Counting id overlap instead diverges from the
standard whenever distances tie or vectors are duplicated, and silently reports
a lower number than every published comparison.

Two consequences of that definition, both invariants
(``docs/methodology/MEASUREMENT-INTEGRITY.md`` I1-I4):

The oracle sees what the system sees. Vector columns are float4, so ground
truth rounds to float32 and only then computes in float64. Keeping full
precision in the oracle makes near-ties disagree for reasons that have nothing
to do with the index.

Out-of-range neighbour ids fail loudly, because NumPy would happily wrap a
negative index and produce ground truth for the wrong vector.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.errors import ConfigError, ErrorContext, Phase

DEFAULT_EPS: Final[float] = 1e-3

FloatArray = npt.NDArray[np.floating[Any]]
IntArray = npt.NDArray[np.integer[Any]]


def _as_oracle_input(vectors: FloatArray) -> npt.NDArray[np.float64]:
    """Round to float32 storage precision, then widen for the arithmetic."""
    return np.ascontiguousarray(vectors, dtype=np.float32).astype(np.float64)


def pairwise_distances(
    corpus: FloatArray, queries: FloatArray, metric: str = "l2"
) -> npt.NDArray[np.float64]:
    """Distance from every query to every corpus vector, smallest is nearest."""
    left = _as_oracle_input(corpus)
    right = _as_oracle_input(queries)
    if left.ndim != 2 or right.ndim != 2:
        raise ConfigError(
            f"expected 2-D corpus and queries, got {corpus.shape} and {queries.shape}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if left.shape[1] != right.shape[1]:
        raise ConfigError(
            f"dimension mismatch: corpus has {left.shape[1]}, queries have {right.shape[1]}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )

    if metric == "l2":
        # Squared L2 throughout: monotone in L2, and skipping the square root
        # avoids introducing rounding the system under test never performs.
        corpus_sq = np.einsum("ij,ij->i", left, left)
        query_sq = np.einsum("ij,ij->i", right, right)
        cross = right @ left.T
        distances = query_sq[:, None] + corpus_sq[None, :] - 2.0 * cross
        return np.asarray(np.maximum(distances, 0.0), dtype=np.float64)
    if metric == "ip":
        return np.asarray(-(right @ left.T), dtype=np.float64)
    if metric == "cosine":
        corpus_norm = np.linalg.norm(left, axis=1)
        query_norm = np.linalg.norm(right, axis=1)
        denominator = query_norm[:, None] * corpus_norm[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = (right @ left.T) / denominator
        # A zero vector has no direction; 1.0 (maximum distance) is the honest
        # answer, and nan would silently sort first or last depending on numpy.
        return np.asarray(
            np.nan_to_num(1.0 - similarity, nan=1.0, posinf=1.0, neginf=1.0), dtype=np.float64
        )
    raise ConfigError(
        f"unknown metric {metric!r}; known metrics: l2, cosine, ip",
        context=ErrorContext(phase=Phase.OFFLINE),
    )


def brute_force_ground_truth(
    corpus: FloatArray, queries: FloatArray, k: int, metric: str = "l2"
) -> tuple[IntArray, npt.NDArray[np.float64]]:
    """Exact nearest neighbours: the oracle every recall figure is measured against."""
    if k < 1:
        raise ConfigError(
            f"k must be at least 1, got {k}", context=ErrorContext(phase=Phase.OFFLINE)
        )
    distances = pairwise_distances(corpus, queries, metric)
    corpus_size = distances.shape[1]
    if k > corpus_size:
        raise ConfigError(
            f"k={k} exceeds the corpus size {corpus_size}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    # Tie-break by id so the oracle is reproducible, exactly as the system is
    # required to be.
    ids = np.lexsort((np.tile(np.arange(corpus_size), (distances.shape[0], 1)), distances), axis=1)
    ids = ids[:, :k]
    ordered = np.take_along_axis(distances, ids, axis=1)
    return ids.astype(np.int64), np.asarray(ordered, dtype=np.float64)


def neighbors_ground_truth(
    corpus: FloatArray,
    queries: FloatArray,
    neighbor_ids: IntArray,
    k: int,
    metric: str = "l2",
) -> npt.NDArray[np.float64]:
    """Ground-truth distances from published neighbour ids, recomputed.

    Large ANN datasets ship both neighbour ids and distances. The distances are
    not used: they were produced by someone else's precision and metric
    convention. Recomputing from the vectors costs one pass instead of a full
    N x Q product, and is the only version we can defend.
    """
    if neighbor_ids.ndim != 2:
        raise ConfigError(
            f"expected 2-D neighbour ids, got shape {neighbor_ids.shape}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if neighbor_ids.shape[0] != queries.shape[0]:
        raise ConfigError(
            f"{neighbor_ids.shape[0]} neighbour rows for {queries.shape[0]} queries",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if k > neighbor_ids.shape[1]:
        raise ConfigError(
            f"k={k} exceeds the {neighbor_ids.shape[1]} published neighbours",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    selected = neighbor_ids[:, :k]
    corpus_size = corpus.shape[0]
    # NumPy would wrap a negative index and index a real vector, producing
    # confident ground truth for the wrong neighbour.
    if selected.min() < 0 or selected.max() >= corpus_size:
        offending = int(selected.min()) if selected.min() < 0 else int(selected.max())
        raise ConfigError(
            f"neighbour id {offending} is outside the corpus of {corpus_size} vectors",
            context=ErrorContext(phase=Phase.OFFLINE),
        )

    left = _as_oracle_input(corpus)
    right = _as_oracle_input(queries)
    gathered = left[selected]
    if metric == "l2":
        deltas = gathered - right[:, None, :]
        return np.asarray(np.einsum("qkd,qkd->qk", deltas, deltas), dtype=np.float64)
    if metric == "ip":
        return np.asarray(-np.einsum("qkd,qd->qk", gathered, right), dtype=np.float64)
    if metric == "cosine":
        query_norm = np.linalg.norm(right, axis=1)[:, None]
        neighbour_norm = np.linalg.norm(gathered, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.einsum("qkd,qd->qk", gathered, right) / (query_norm * neighbour_norm)
        return np.asarray(
            np.nan_to_num(1.0 - similarity, nan=1.0, posinf=1.0, neginf=1.0), dtype=np.float64
        )
    raise ConfigError(f"unknown metric {metric!r}", context=ErrorContext(phase=Phase.OFFLINE))


def recall_at_k(
    true_distances: npt.NDArray[np.float64],
    run_distances: npt.NDArray[np.float64],
    k: int,
    eps: float = DEFAULT_EPS,
) -> float:
    """Recall@k by distance threshold, averaged over queries.

    A returned neighbour counts when its distance is at most the k-th true
    distance plus ``eps``. This is the ANN-Benchmarks definition; id overlap is
    a different, lower number that no published comparison uses.
    """
    if k < 1:
        raise ConfigError(
            f"k must be at least 1, got {k}", context=ErrorContext(phase=Phase.OFFLINE)
        )
    if true_distances.ndim != 2 or run_distances.ndim != 2:
        raise ConfigError(
            "recall needs 2-D distance matrices (queries x neighbours)",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if true_distances.shape[0] != run_distances.shape[0]:
        raise ConfigError(
            f"{true_distances.shape[0]} ground-truth rows for {run_distances.shape[0]} result rows",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if true_distances.shape[1] < k:
        raise ConfigError(
            f"ground truth has {true_distances.shape[1]} neighbours, k={k}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )

    thresholds = true_distances[:, k - 1] + eps
    considered = run_distances[:, :k]
    within = considered <= thresholds[:, None]
    return float(np.mean(within.sum(axis=1) / k))


def recall_from_ids(
    corpus: FloatArray,
    queries: FloatArray,
    returned_ids: IntArray,
    true_distances: npt.NDArray[np.float64],
    k: int,
    metric: str = "l2",
    eps: float = DEFAULT_EPS,
) -> float:
    """Recall for a run that reported ids, by recomputing their true distances.

    Used when the system returns ids without distances, or when its distances
    cannot be trusted to use the same convention as the oracle (TRD D6).
    """
    if returned_ids.shape[1] < k:
        raise ConfigError(
            f"run returned {returned_ids.shape[1]} neighbours, k={k}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    run_distances = neighbors_ground_truth(corpus, queries, returned_ids, k, metric)
    return recall_at_k(true_distances, run_distances, k, eps)


def ndcg_at_k(ranked_ids: list[int], relevance: dict[int, float], k: int) -> float:
    """Normalised discounted cumulative gain, graded relevance, log2 discount."""
    if k < 1:
        raise ConfigError(
            f"k must be at least 1, got {k}", context=ErrorContext(phase=Phase.OFFLINE)
        )
    gains = [relevance.get(doc_id, 0.0) for doc_id in ranked_ids[:k]]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(np.asarray(gains) * discounts))

    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    if not ideal_gains or max(ideal_gains, default=0.0) <= 0.0:
        # No relevant document exists for this query; nDCG is undefined rather
        # than zero, and callers aggregate over queries that have judgements.
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float(np.sum(np.asarray(ideal_gains) * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_n(ranked_ids: list[int], relevant: set[int], n: int) -> float:
    """Fraction of the relevant set retrieved in the top n."""
    if not relevant:
        return 0.0
    hits = len(set(ranked_ids[:n]) & relevant)
    return hits / len(relevant)


def mrr_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    """Reciprocal rank of the first relevant document, 0 when none appears."""
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def success_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    """1.0 when any relevant document appears in the top k."""
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0
