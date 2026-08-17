"""Pairing two systems' per-query samples so a paired test can be run on them.

Invariant I14 requires a paired test to compare two systems, and a paired test
requires paired samples: element *i* of each sequence must be the same query
answered by a different system. Producing that pairing is the whole job of this
module, and refusing to produce it is half of it.

The refusal matters more than it looks. A latency list skips queries that errored
or timed out, so position *i* in the list is not query *i*. Pairing by position
would silently misalign every sample after the first timeout and yield a
confident, wrong p-value — a comparison that looks more rigorous than the median
table it replaced while being less true. Samples are therefore keyed by query id,
and differing key sets are a typed error rather than an intersection (I22:
"differing qid sets are a typed error, never a silent partial comparison").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from theodb_bench.errors import ConfigError, ErrorContext, Phase


@dataclass(frozen=True)
class PairedSamples:
    """Two systems' values for the same queries, in the same order."""

    query_ids: tuple[int, ...]
    a: tuple[float, ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.query_ids) == len(self.a) == len(self.b)):
            raise ConfigError(
                "paired samples must have one value per query on each side",
                context=ErrorContext(phase=Phase.OFFLINE),
            )


def pair_by_query(system_a: Mapping[int, float], system_b: Mapping[int, float]) -> PairedSamples:
    """Pair two systems' per-query values, or refuse.

    Refuses on a differing query set rather than intersecting it. An intersection
    would compare the two systems on the subset where both happened to succeed,
    which is a different and easier question than the one being asked — and it
    would quietly favour whichever system failed on its hardest queries.
    """
    ids_a, ids_b = set(system_a), set(system_b)
    if not ids_a or not ids_b:
        raise ConfigError(
            "no paired samples: at least one system reported no per-query values, "
            "so there is nothing to pair and nothing to test",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)[:5]
        only_b = sorted(ids_b - ids_a)[:5]
        raise ConfigError(
            f"query sets differ: {len(ids_a)} vs {len(ids_b)} queries, "
            f"{len(ids_a ^ ids_b)} not in both (first only in A: {only_a}; "
            f"first only in B: {only_b}). Comparing the intersection would test "
            f"the subset where both systems succeeded, which is an easier question "
            f"than the one asked and favours whichever system dropped its hardest "
            f"queries.",
            context=ErrorContext(phase=Phase.OFFLINE),
        )

    ordered = tuple(sorted(ids_a))
    return PairedSamples(
        query_ids=ordered,
        a=tuple(float(system_a[q]) for q in ordered),
        b=tuple(float(system_b[q]) for q in ordered),
    )


@dataclass(frozen=True)
class RecallMatch:
    """Two operating points read at the same quality, one from each engine."""

    label_a: str
    recall_a: float
    label_b: str
    recall_b: float

    @property
    def gap(self) -> float:
        return abs(self.recall_a - self.recall_b)


def match_by_recall(
    recalls_a: Mapping[str, float],
    recalls_b: Mapping[str, float],
    *,
    tolerance: float = 0.01,
) -> RecallMatch | None:
    """The closest pair of operating points at comparable quality, or None.

    Two engines never share a configuration label: the label carries
    engine-specific parameters, `probes=20` on one side and
    `num_leaves_to_search=20` on the other. Two knobs named differently and set to
    the same integer are not the same operating point, and pairing them would
    compare the knobs rather than the engines.

    Quality is the axis both engines share, so the frontiers are read at matched
    recall. `tolerance` is honoured rather than taking the nearest pair at any
    distance: frontiers that never meet -- one topping out below the other's floor,
    which is what an unrescored quantizer produces -- have no comparable point, and
    inventing one would compare a fast low-quality configuration against a slow
    high-quality one and call the first a winner.
    """
    best: RecallMatch | None = None
    for label_a, recall_a in recalls_a.items():
        for label_b, recall_b in recalls_b.items():
            candidate = RecallMatch(label_a, float(recall_a), label_b, float(recall_b))
            if candidate.gap <= tolerance and (best is None or candidate.gap < best.gap):
                best = candidate
    return best


def render_paired_verdict(
    name_a: str,
    samples_a: Mapping[int, float],
    name_b: str,
    samples_b: Mapping[int, float],
    *,
    metric: str,
    lower_is_better: bool = True,
) -> str:
    """One line of verdict for a paired comparison, or the reason there is none.

    Two medians printed in adjacent columns are two summaries near each other, not
    a comparison of two systems, and a reader will infer a winner from them
    whether or not one exists. This says whether the difference survives a paired
    test, in which direction, by how much, and with what confidence — or says the
    runs cannot be paired and stops.

    Refusing is the important half. Falling back to medians when the pairing fails
    would put a comparison in front of a reader who has no way to know it is not
    one.
    """
    try:
        paired = pair_by_query(samples_a, samples_b)
    except ConfigError as exc:
        return f"**{name_a} vs {name_b}** — not comparable: {exc.args[0].splitlines()[0]}"

    from theodb_bench.analysis.significance import compare_systems

    result = compare_systems(list(paired.a), list(paired.b))
    effect = result.effect
    n = len(paired.query_ids)

    if not result.significant:
        return (
            f"**{name_a} vs {name_b}** ({metric}) — **indistinguishable** "
            f"(p = {result.p_randomisation:.4f}, n = {n}, "
            f"95% CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}])"
        )

    a_is_better = (effect.mean_difference < 0) if lower_is_better else (effect.mean_difference > 0)
    faster, slower = (name_a, name_b) if a_is_better else (name_b, name_a)

    # `wins` counts the queries where A's value was larger. For a metric where
    # lower is better that is where A was *slower*, so printing it beside "A
    # beats B" reads as A losing on most queries. Counted in the direction just
    # named instead.
    a_larger, b_larger = effect.wins, effect.losses
    if lower_is_better:
        better_count = b_larger if a_is_better else a_larger
    else:
        better_count = a_larger if a_is_better else b_larger

    dz = f", dz = {effect.cohens_dz:.2f}" if effect.cohens_dz is not None else ""
    return (
        f"**{faster}** beats **{slower}** on {metric} "
        f"(p = {result.p_randomisation:.4f}, n = {n}, "
        f"95% CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}], "
        f"mean diff = {effect.mean_difference:+.3f}{dz}; "
        f"{faster} faster on {better_count} of {n} queries, {effect.ties} tied)"
    )
