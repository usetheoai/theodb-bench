"""Rank fusion.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) combines
rankings by position rather than by score, which is what makes it usable across
legs whose scores are not on a comparable scale -- a BM25 score and a cosine
distance have no common unit, and normalising them into one invents a
relationship nobody measured.

This is the offline twin of whatever the database does internally. Having both
is the point: a fusion computed here from the individual legs can be compared
against the system's own fused output, and a divergence is a finding rather
than a mystery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

DEFAULT_K: Final[int] = 60
"""The constant from the original paper. It damps the contribution of top
ranks; changing it changes the fusion, so it is recorded rather than assumed."""


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked id lists into one ranking.

    ``rankings`` maps a leg name to its ranked document ids, best first. The
    score of a document is the sum over legs of ``weight / (k + rank)``, with
    ranks starting at 1.

    Ties are broken by document id, so the output is reproducible. Without
    that, two documents with identical fused scores would come out in whatever
    order the sort happened to produce.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: dict[int, float] = {}
    for leg, ranked in rankings.items():
        weight = 1.0 if weights is None else weights.get(leg, 1.0)
        if weight == 0.0:
            continue
        seen: set[int] = set()
        for position, doc_id in enumerate(ranked, start=1):
            # A leg that returns the same document twice must not be counted
            # twice: that would let a buggy leg outvote a correct one.
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def fuse_to_ids(
    rankings: Mapping[str, Sequence[int]],
    n: int,
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
) -> list[int]:
    """Fused ranking truncated to the top ``n`` ids."""
    return [doc_id for doc_id, _ in reciprocal_rank_fusion(rankings, k, weights)[:n]]


def rank_agreement(left: Sequence[int], right: Sequence[int], n: int) -> float:
    """Overlap between two rankings at depth ``n``.

    Used to compare a fusion computed here against the system's own, and to
    quantify how much a rerank moved the ranking. Not a quality metric: two
    rankings can agree perfectly and both be wrong.
    """
    if n <= 0:
        raise ValueError(f"depth must be positive, got {n}")
    top_left, top_right = set(left[:n]), set(right[:n])
    if not top_left and not top_right:
        return 1.0
    return len(top_left & top_right) / max(len(top_left), len(top_right), 1)
