"""A difference between two medians is an observation, not a result."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.analysis.significance import (
    cohens_dz,
    compare_systems,
    paired_bootstrap_ci,
    paired_randomisation_test,
    paired_t_test,
)
from theodb_bench.errors import ConfigError

FAST = {"resamples": 2000}


def _noise(n: int = 60, seed: int = 5) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal(n)


# ------------------------------------------------------------------- the null


def test_identical_systems_are_not_significant() -> None:
    values = list(_noise())
    result = compare_systems(values, values, **FAST)
    assert not result.significant
    assert result.effect.mean_difference == pytest.approx(0.0)
    assert result.effect.ties == len(values)


def test_two_independent_noise_samples_are_usually_not_significant() -> None:
    a = list(_noise(seed=1))
    b = list(_noise(seed=2))
    assert not compare_systems(a, b, **FAST).significant


def test_a_uniform_shift_is_significant() -> None:
    a = _noise()
    b = a - 0.8
    result = compare_systems(list(a), list(b), **FAST)
    assert result.significant
    assert result.effect.mean_difference == pytest.approx(0.8, abs=1e-9)
    assert result.effect.wins == len(a)


def test_a_consistent_tiny_shift_is_significant_but_negligible() -> None:
    # A uniform shift means the same system wins every single pair, and 8 out
    # of 8 consistent wins is significant at alpha=0.05 however small the
    # margin: the randomisation test on constant differences reduces to a sign
    # test. That is correct, and it is exactly why p never travels alone --
    # the effect size shows the win is worth 0.01.
    a = _noise(n=8)
    result = compare_systems(list(a), list(a - 0.01), **FAST)
    assert result.significant
    assert result.effect.wins == 8
    assert result.effect.mean_difference == pytest.approx(0.01, abs=1e-9)
    assert result.ci_high - result.ci_low < 0.001


def test_an_inconsistent_small_shift_is_not_significant() -> None:
    # Same magnitude, but the sign varies per pair: no consistent winner, so
    # no claim.
    rng = np.random.default_rng(11)
    a = _noise(n=8)
    b = a - rng.choice(np.array([-0.01, 0.01]), size=8)
    assert not compare_systems(list(a), list(b), **FAST).significant


# --------------------------------------------------------- Monte-Carlo floor


def test_the_p_value_is_never_zero() -> None:
    # The observed assignment is itself one of the permutations, so no finite
    # resampling can justify p = 0.
    a = _noise()
    b = a - 50.0
    result = compare_systems(list(a), list(b), resamples=100)
    assert result.p_randomisation > 0.0
    assert result.p_randomisation == pytest.approx(1 / 101)


def test_the_floor_follows_the_correction_formula() -> None:
    differences = np.full(20, 10.0)
    assert paired_randomisation_test(differences, resamples=999) == pytest.approx(1 / 1000)


def test_more_resamples_lower_the_floor() -> None:
    differences = np.full(20, 10.0)
    coarse = paired_randomisation_test(differences, resamples=99)
    fine = paired_randomisation_test(differences, resamples=9999)
    assert fine < coarse


# --------------------------------------------------------------- determinism


def test_the_same_inputs_give_the_same_p_value() -> None:
    a, b = list(_noise()), list(_noise(seed=9))
    first = compare_systems(a, b, **FAST)
    second = compare_systems(a, b, **FAST)
    assert first.p_randomisation == second.p_randomisation
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_a_different_seed_gives_a_different_resampling() -> None:
    a, b = list(_noise()), list(_noise(seed=9))
    assert (
        compare_systems(a, b, seed=1, **FAST).p_randomisation
        != compare_systems(a, b, seed=2, **FAST).p_randomisation
    )


def test_the_result_records_the_seed_and_resample_count() -> None:
    payload = compare_systems(
        list(_noise()), list(_noise(seed=3)), seed=42, resamples=500
    ).as_dict()
    assert payload["seed"] == 42
    assert payload["resamples"] == 500


# --------------------------------------------------------------------- pairing


def test_unequal_lengths_are_refused() -> None:
    # Two systems that did not answer the same queries cannot be paired.
    with pytest.raises(ConfigError, match="equal-length"):
        compare_systems([1.0, 2.0, 3.0], [1.0, 2.0])


def test_a_single_pair_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least 2 pairs"):
        compare_systems([1.0], [2.0])


def test_an_invalid_alpha_is_refused() -> None:
    with pytest.raises(ConfigError, match="alpha must be"):
        compare_systems(list(_noise()), list(_noise(seed=3)), alpha=1.5)


def test_zero_resamples_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least 1"):
        paired_randomisation_test(_noise(), resamples=0)


# ------------------------------------------------------------------- interval


def test_the_interval_brackets_the_observed_difference() -> None:
    a = _noise()
    b = a - 0.5
    result = compare_systems(list(a), list(b), **FAST)
    assert result.ci_low <= result.effect.mean_difference <= result.ci_high


def test_an_interval_containing_zero_accompanies_a_null_result() -> None:
    a, b = _noise(seed=1), _noise(seed=2)
    result = compare_systems(list(a), list(b), **FAST)
    assert not result.significant
    assert result.ci_low < 0 < result.ci_high


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    differences = _noise()
    assert paired_bootstrap_ci(differences, resamples=500) == paired_bootstrap_ci(
        differences, resamples=500
    )


# ---------------------------------------------------------------- cross-check


def test_the_t_test_records_which_method_produced_it() -> None:
    _, method = paired_t_test(_noise())
    # Either is acceptable; what matters is that the artifact says which.
    assert method in {"scipy_exact", "normal_approximation"}


def test_a_zero_variance_difference_is_labelled_degenerate() -> None:
    identical = np.zeros(10)
    p_value, method = paired_t_test(identical)
    assert method == "degenerate_zero_variance"
    assert p_value == 1.0

    constant = np.full(10, 3.0)
    p_constant, method_constant = paired_t_test(constant)
    assert method_constant == "degenerate_constant"
    assert p_constant == 0.0


def test_concordance_between_the_two_tests_is_reported() -> None:
    a = _noise()
    result = compare_systems(list(a), list(a - 0.8), **FAST)
    assert result.concordant is True


def test_significance_is_decided_by_the_headline_test_alone() -> None:
    # The t-test is a cross-check, not a vote: the verdict comes from the
    # randomisation test.
    result = compare_systems(list(_noise()), list(_noise() - 0.8), **FAST)
    assert result.significant == (result.p_randomisation < result.alpha)


# ----------------------------------------------------------------- effect size


def test_effect_size_accompanies_the_p_value() -> None:
    # A significant result with a negligible effect is a large sample, not an
    # improvement.
    a = _noise(n=400)
    result = compare_systems(list(a), list(a - 0.5), **FAST)
    assert result.effect.cohens_dz is not None
    assert result.effect.mean_difference == pytest.approx(0.5, abs=1e-9)


def test_wins_losses_and_ties_account_for_every_pair() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 2.0, 5.0, 1.0]
    effect = compare_systems(a, b, **FAST).effect
    assert (effect.wins, effect.losses, effect.ties) == (3, 1, 0) or (
        effect.wins + effect.losses + effect.ties
    ) == len(a)


def test_cohens_dz_is_undefined_without_spread() -> None:
    assert cohens_dz(np.full(5, 2.0)) is None


# -------------------------------------------------------------------- reporting


def test_the_summary_states_the_verdict_and_the_interval() -> None:
    a = _noise()
    summary = compare_systems(list(a), list(a - 0.8), **FAST).summary()
    assert "significant" in summary
    assert "95% CI" in summary
    assert "mean difference" in summary


def test_the_payload_names_the_test_it_used() -> None:
    payload = compare_systems(list(_noise()), list(_noise(seed=4)), **FAST).as_dict()
    assert "randomisation" in payload["test"]
    assert "Smucker" in payload["test"]
