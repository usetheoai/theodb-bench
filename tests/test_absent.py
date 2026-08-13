"""A missing measurement must never be mistakable for a measured zero."""

from __future__ import annotations

import json

import pytest
from theodb_bench.absent import (
    AbsenceReason,
    Absent,
    encode,
    invalid,
    is_present,
    not_collected,
    unavailable,
    unsupported,
)


def test_absent_is_not_equal_to_zero() -> None:
    # mypy already proves these types cannot compare equal; the runtime
    # assertion stays because the whole point of this module is that a future
    # __eq__ or a coercion helper must never make an absence look like a zero.
    assert unavailable() != 0  # type: ignore[comparison-overlap]
    assert not is_present(unavailable())


def test_measured_zero_is_present() -> None:
    assert is_present(0)
    assert is_present(0.0)


@pytest.mark.parametrize(
    ("factory", "reason"),
    [
        (unsupported, AbsenceReason.UNSUPPORTED),
        (unavailable, AbsenceReason.UNAVAILABLE),
        (not_collected, AbsenceReason.NOT_COLLECTED),
        (invalid, AbsenceReason.INVALID),
    ],
)
def test_each_reason_round_trips_through_json(factory: type, reason: AbsenceReason) -> None:
    value = factory("perf event not exposed by kernel")
    payload = json.loads(json.dumps(encode(value)))
    assert payload == {
        "absent": reason.value,
        "detail": "perf event not exposed by kernel",
    }


def test_encode_passes_real_measurements_through_unchanged() -> None:
    assert encode(0) == 0
    assert encode(12.5) == 12.5


def test_absence_reasons_are_distinct() -> None:
    # They are not interchangeable: "the system cannot do this" and "we chose
    # not to collect it" lead to different conclusions about a result.
    reasons = {unsupported(), unavailable(), not_collected(), invalid()}
    assert len(reasons) == 4


def test_absent_is_frozen() -> None:
    value = unavailable()
    with pytest.raises(AttributeError):
        value.reason = AbsenceReason.INVALID  # type: ignore[misc]


def test_str_includes_detail_when_present() -> None:
    assert str(unavailable()) == "unavailable"
    assert str(unavailable("no cgroup v2")) == "unavailable: no cgroup v2"


def test_encode_omits_detail_when_absent_has_none() -> None:
    assert encode(Absent(AbsenceReason.NOT_COLLECTED)) == {"absent": "not_collected"}
