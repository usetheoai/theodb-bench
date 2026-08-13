"""A profile declares what a result may be used for, not how fast it runs."""

from __future__ import annotations

import pytest
from theodb_bench.errors import ConfigError
from theodb_bench.profiles import PROFILES, ProfileName, get_profile


def test_every_declared_profile_exists() -> None:
    assert set(PROFILES) == set(ProfileName)


def test_only_release_is_publishable() -> None:
    publishable = {name.value for name, p in PROFILES.items() if p.publishable}
    assert publishable == {"release"}


def test_release_requires_frozen_methodology_and_a_clean_tree() -> None:
    release = get_profile("release")
    assert release.frozen_methodology
    assert release.dirty_tree_invalidates
    assert release.isolation_required
    assert release.preflight_required


def test_smoke_and_research_do_not_gate_regressions() -> None:
    # Neither runs on controlled hardware, so a delta between two of their runs
    # is an observation, not a signal.
    assert not get_profile("smoke").regression_gate
    assert not get_profile("research").regression_gate


def test_gated_profiles_require_repetitions() -> None:
    for name in ("pr", "nightly", "release"):
        assert get_profile(name).min_repetitions >= 3


def test_repetition_floor_is_enforced() -> None:
    with pytest.raises(ConfigError, match="at least 5"):
        get_profile("release").require_repetitions(3)


def test_repetition_floor_accepts_the_exact_minimum() -> None:
    get_profile("pr").require_repetitions(3)


def test_unknown_profile_is_rejected_by_name() -> None:
    with pytest.raises(ConfigError, match="unknown profile"):
        get_profile("production")


def test_profiles_are_immutable() -> None:
    with pytest.raises(AttributeError):
        get_profile("smoke").publishable = True  # type: ignore[misc]
