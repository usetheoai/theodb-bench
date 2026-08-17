"""Connecting the regression module to a run.

`analysis/regression.py` implemented comparability checks, per-metric gates and a
verdict, and had zero importers in `src/`. Three profiles declared
`regression_gate = True` while nothing compared anything, which is a promise the
contract made and the code did not keep.

The default gates are deliberately conservative and their source is recorded as
advisory: this project's own rule is that a regression threshold is only
trustworthy after the runner's own variance has been measured, and that
measurement does not exist yet. An advisory threshold that says so is honest; a
number presented as measured would not be.
"""

from __future__ import annotations

from typing import Any

from theodb_bench.analysis.regression import (
    ADVISORY_SOURCE,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    BaselineRef,
    Candidate,
    Gate,
    check_comparability,
    compare,
)

#: Advisory until the runner's own variance is measured. Same-day evidence for
#: why they are not tighter: the same configuration re-run on the same host
#: varied by 24% and 46% in median throughput.
DEFAULT_GATES: tuple[Gate, ...] = (
    Gate(
        metric="throughput_per_second",
        direction=HIGHER_IS_BETTER,
        max_regression_pct=10.0,
        threshold_source=ADVISORY_SOURCE,
    ),
    Gate(
        metric="latency_p95_ms",
        direction=LOWER_IS_BETTER,
        max_regression_pct=15.0,
        threshold_source=ADVISORY_SOURCE,
    ),
)


def regression_for(
    candidate: dict[str, Any], baseline: dict[str, Any] | None
) -> tuple[dict[str, Any], bool]:
    """The regression artefact for a run, and whether the baseline was comparable.

    The boolean is what the profile gate reads. It is separate from the verdict
    because "these two runs may not be compared" and "the comparison found a
    regression" send an operator to different places.
    """
    candidate_ref = Candidate(**candidate)
    baseline_ref = BaselineRef(**baseline) if baseline is not None else None
    comparable, _ = check_comparability(candidate_ref, baseline_ref)
    payload = compare(candidate_ref, baseline_ref, list(DEFAULT_GATES))
    return payload, comparable
