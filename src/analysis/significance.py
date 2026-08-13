"""Paired significance testing.

A difference between two medians is an observation. Turning it into a claim
requires a test, and the test has to be **paired**: the two systems answered the
same queries, so the per-query differences carry information that comparing two
aggregates throws away.

The headline test is a paired randomisation (permutation) test, following
Smucker, Allan & Carterette (CIKM 2007), which compared significance tests for
IR evaluation and recommended it. Wilcoxon and the sign test are deliberately
not used: they discard magnitude and handle ties poorly.

Reported alongside it, per Urbano, Marrero & Martín (SIGIR 2013): a paired
percentile bootstrap confidence interval and a paired t-test as a concordant
cross-check. Three procedures that agree are worth more than one that is
asserted.

Two properties are non-negotiable:

**Monte-Carlo correction.** ``p = (count + 1) / (B + 1)``. The observed
assignment is itself one of the permutations, so an uncorrected estimate can
report ``p = 0``, which is not a probability any finite resampling can justify.

**A fixed seed.** The same inputs must give the same ``p`` and the same
interval, or the result is not reproducible.

Everything here is pure NumPy on purpose. Permutation and bootstrap are two
trivial resamplings, and taking a SciPy dependency for them would buy a
compatibility surface without buying correctness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.errors import ConfigError, ErrorContext, Phase

DEFAULT_RESAMPLES: Final[int] = 10_000
DEFAULT_SEED: Final[int] = 20260813
DEFAULT_ALPHA: Final[float] = 0.05

Differences = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EffectSize:
    """How large the difference is, separate from how sure we are it exists.

    A significant result with a negligible effect is a large sample, not an
    improvement, and reporting ``p`` without this invites that confusion.
    """

    mean_difference: float
    cohens_dz: float | None
    wins: int
    losses: int
    ties: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_difference": self.mean_difference,
            "cohens_dz": self.cohens_dz,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
        }


@dataclass(frozen=True)
class SignificanceResult:
    """The full verdict: three procedures, an effect size, and the inputs."""

    n: int
    resamples: int
    seed: int
    alpha: float
    p_randomisation: float
    p_ttest: float
    p_ttest_method: str
    ci_low: float
    ci_high: float
    effect: EffectSize
    concordant: bool
    """Whether the randomisation test and the t-test agree at ``alpha``."""

    @property
    def significant(self) -> bool:
        """Significance is decided by the headline test alone."""
        return self.p_randomisation < self.alpha

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "resamples": self.resamples,
            "seed": self.seed,
            "alpha": self.alpha,
            "p_randomisation": self.p_randomisation,
            "p_ttest": self.p_ttest,
            "p_ttest_method": self.p_ttest_method,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "significant": self.significant,
            "concordant": self.concordant,
            "effect": self.effect.as_dict(),
            "test": "paired randomisation (Smucker, Allan & Carterette, CIKM 2007)",
        }

    def summary(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        agreement = "" if self.concordant else "; the t-test disagrees, so treat with care"
        return (
            f"{verdict} at alpha={self.alpha} (p={self.p_randomisation:.4g}, n={self.n}), "
            f"mean difference {self.effect.mean_difference:+.6g}, "
            f"95% CI [{self.ci_low:.6g}, {self.ci_high:.6g}]{agreement}"
        )


def _paired_differences(system_a: Sequence[float], system_b: Sequence[float]) -> Differences:
    if len(system_a) != len(system_b):
        raise ConfigError(
            f"paired testing needs equal-length samples; got {len(system_a)} and "
            f"{len(system_b)}. Two systems that did not answer the same queries "
            "cannot be paired.",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if len(system_a) < 2:
        raise ConfigError(
            f"paired testing needs at least 2 pairs, got {len(system_a)}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    return np.asarray(
        np.asarray(system_a, dtype=np.float64) - np.asarray(system_b, dtype=np.float64),
        dtype=np.float64,
    )


def paired_randomisation_test(
    differences: Differences, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> float:
    """Two-sided p-value from sign-flipping the paired differences.

    Under the null hypothesis the two systems are interchangeable per query, so
    the sign of each difference is arbitrary. Enumerating random sign
    assignments gives the distribution of the mean difference under that null.
    """
    if resamples < 1:
        raise ConfigError(
            f"resamples must be at least 1, got {resamples}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    observed = float(np.abs(np.mean(differences)))
    rng = np.random.default_rng(seed)
    n = differences.size

    # Sign flips in one vectorised pass rather than a Python loop.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, n))
    means = np.abs((signs * differences).mean(axis=1))
    count = int(np.count_nonzero(means >= observed))

    # Monte-Carlo correction: the observed assignment is one of the
    # permutations, so p is never 0.
    return (count + 1) / (resamples + 1)


def paired_bootstrap_ci(
    differences: Differences,
    alpha: float = DEFAULT_ALPHA,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference."""
    rng = np.random.default_rng(seed)
    n = differences.size
    indices = rng.integers(0, n, size=(resamples, n))
    means = differences[indices].mean(axis=1)
    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return low, high


def paired_t_test(differences: Differences) -> tuple[float, str]:
    """Two-sided paired t-test.

    Returns the p-value and the method used to obtain it. SciPy is optional
    here; without it a normal approximation is used and the artifact records
    which one produced the number, rather than letting a reader assume the
    exact distribution was used.
    """
    n = differences.size
    mean = float(np.mean(differences))
    stdev = float(np.std(differences, ddof=1))
    if stdev == 0:
        # Every pair differs identically. Either the systems are identical
        # (mean 0) or they differ by a constant with no variance at all.
        return (1.0, "degenerate_zero_variance") if mean == 0 else (0.0, "degenerate_constant")

    t_statistic = mean / (stdev / math.sqrt(n))
    try:
        from scipy import stats

        p_value = float(2 * stats.t.sf(abs(t_statistic), df=n - 1))
        return p_value, "scipy_exact"
    except ImportError:
        # Normal approximation. Conservative to state, and the label says so.
        p_value = float(math.erfc(abs(t_statistic) / math.sqrt(2)))
        return p_value, "normal_approximation"


def cohens_dz(differences: Differences) -> float | None:
    """Standardised effect size for paired samples."""
    stdev = float(np.std(differences, ddof=1))
    if stdev == 0:
        return None
    return float(np.mean(differences)) / stdev


def compare_systems(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> SignificanceResult:
    """Decide whether system A differs from system B on a paired measurement.

    Both sequences must be per-query values in the same order: element *i* of
    each is the same query answered by a different system. Aggregates cannot be
    paired, and passing them here would silently test the wrong thing.
    """
    if not 0.0 < alpha < 1.0:
        raise ConfigError(
            f"alpha must be in (0, 1), got {alpha}", context=ErrorContext(phase=Phase.OFFLINE)
        )

    differences = _paired_differences(system_a, system_b)
    p_randomisation = paired_randomisation_test(differences, resamples, seed)
    low, high = paired_bootstrap_ci(differences, alpha, resamples, seed)
    p_ttest, method = paired_t_test(differences)

    wins = int(np.count_nonzero(differences > 0))
    losses = int(np.count_nonzero(differences < 0))
    ties = int(np.count_nonzero(differences == 0))

    return SignificanceResult(
        n=differences.size,
        resamples=resamples,
        seed=seed,
        alpha=alpha,
        p_randomisation=p_randomisation,
        p_ttest=p_ttest,
        p_ttest_method=method,
        ci_low=low,
        ci_high=high,
        effect=EffectSize(
            mean_difference=float(np.mean(differences)),
            cohens_dz=cohens_dz(differences),
            wins=wins,
            losses=losses,
            ties=ties,
        ),
        concordant=(p_randomisation < alpha) == (p_ttest < alpha),
    )
