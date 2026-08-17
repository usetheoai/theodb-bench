"""A profile declares what a result may be used for, not how fast it runs."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

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


# ------------------------------------------ every rigour flag has a consumer
#
# Measured 2026-08-17: of the four flags a profile declares, `publishable` and
# `dirty_tree_invalidates` were enforced and `regression_gate` and
# `frozen_methodology` were not. All five references to `regression_gate` were its
# own definition, and `frozen_methodology` was echoed by `list` and nowhere else.
#
# A profile is the contract that separates local validation from a publishable
# number, and it is what an outside reader consults to know what a number is
# worth. A decorative flag in that contract is the same defect as an invariant
# documented and never executed.


def _flags_used_as_gates(source: Path) -> set[str]:
    """Flag names this file uses to *decide* something, not merely to print it.

    Textual presence is not consumption. `regression_gate` and
    `frozen_methodology` both appeared in `cli.py` -- as values echoed by the
    `list` command, which decides nothing. A flag is a gate when it steers
    control flow (an `if`, a boolean operator, a negation) or when it is passed as
    the `required=` argument that makes a validation check blocking.
    """
    tree = ast.parse(source.read_text())
    used: set[str] = set()

    def names_in(node: ast.AST) -> set[str]:
        return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            used |= names_in(node.test)
        elif isinstance(node, (ast.BoolOp, ast.Compare)):
            used |= names_in(node)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            used |= names_in(node.operand)
        elif isinstance(node, ast.IfExp):
            used |= names_in(node.test)
        elif isinstance(node, ast.keyword) and node.arg == "required":
            used |= names_in(node.value)
    return used


def test_every_profile_flag_steers_something() -> None:
    """A profile is the contract that says what a number is worth. A flag in it
    that no code reads is a promise nobody keeps."""
    src = Path(__file__).resolve().parent.parent / "src"
    definition = src / "profiles.py"
    release = PROFILES["release"]
    flags = {
        f.name for f in dataclasses.fields(release) if isinstance(getattr(release, f.name), bool)
    }

    gated: set[str] = set()
    for path in src.rglob("*.py"):
        if path != definition:
            gated |= _flags_used_as_gates(path)

    unconsumed = sorted(flags - gated)
    assert not unconsumed, (
        f"profile flag(s) declared and never used to decide anything: {unconsumed}. "
        f"Appearing in a printed listing is not consumption."
    )
