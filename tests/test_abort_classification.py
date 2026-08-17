"""Why a run aborted, kept distinct from whether the system crashed.

The runner used to set `sut_crashed = True` for any exception, so three
different facts reached the report as one sentence: "system under test crashed
during the run". Measured on 2026-08-17, all three happened in a single session:

  * the knob gate refused a run because a requested search parameter was not in
    force -- the harness working exactly as designed;
  * a `CREATE INDEX` on a million vectors was cancelled by the harness's own 60 s
    `statement_timeout` -- a budget the harness set, not a fault of the server;
  * nothing, in fact, crashed: the container was `Up (healthy)` and its log held
    no PANIC or FATAL.

For a benchmark whose results are published, that misattribution is the
expensive kind: a reader who sees "the system under test crashed" concludes the
database is unstable. All three still invalidate the run -- no number may be
published from an aborted run -- but they must say what happened.
"""

from __future__ import annotations

import pytest
from theodb_bench.abort import AbortKind, classify_abort
from theodb_bench.errors import AdapterError, ErrorContext, Phase, SystemUnavailableError

psycopg = pytest.importorskip("psycopg")
pg_errors = pytest.importorskip("psycopg.errors")


def test_a_gate_refusal_is_a_refusal_not_a_crash() -> None:
    exc = AdapterError(
        "theodb cannot apply search parameter 'num_leaves_to_search'",
        context=ErrorContext(phase=Phase.MEASUREMENT, system="theodb"),
    )

    assert classify_abort(exc) is AbortKind.REFUSED


def test_a_cancelled_statement_is_a_budget_overrun_not_a_crash() -> None:
    """Measured: `CREATE INDEX` on 1M x 128 exceeded the 60 s statement_timeout."""
    exc = pg_errors.QueryCanceled("canceling statement due to statement timeout")

    assert classify_abort(exc) is AbortKind.BUDGET_EXCEEDED


def test_an_unreachable_server_is_a_crash() -> None:
    exc = SystemUnavailableError(
        "could not connect to theodb",
        context=ErrorContext(phase=Phase.BOOTSTRAP, system="theodb"),
    )

    assert classify_abort(exc) is AbortKind.CRASHED


def test_a_dropped_connection_is_a_crash() -> None:
    exc = psycopg.OperationalError("server closed the connection unexpectedly")

    assert classify_abort(exc) is AbortKind.CRASHED


def test_cancellation_is_checked_before_connection_loss() -> None:
    """The ordering this classifier rests on, pinned against the real driver.

    `psycopg.errors.QueryCanceled` is a subclass of `psycopg.OperationalError`.
    A classifier that tested the connection-lost family first would call every
    cancelled statement a crash -- the exact misattribution being removed, so the
    property is asserted rather than assumed.
    """
    assert issubclass(pg_errors.QueryCanceled, psycopg.OperationalError)
    assert classify_abort(pg_errors.QueryCanceled("cancelled")) is AbortKind.BUDGET_EXCEEDED


def test_an_unrecognised_failure_is_reported_as_a_crash() -> None:
    """The conservative default: an unclassified abort is the worst case.

    Guessing 'refused' for something unrecognised would let a real crash be
    reported as the harness declining to measure, which is the failure this
    classification exists to prevent -- pointed the other way.
    """
    assert classify_abort(RuntimeError("something nobody anticipated")) is AbortKind.CRASHED


@pytest.mark.parametrize("kind", list(AbortKind))
def test_every_kind_invalidates_the_run(kind: AbortKind) -> None:
    """No number may be published from an aborted run, whatever the reason."""
    assert kind.invalidates is True


@pytest.mark.parametrize("kind", list(AbortKind))
def test_every_kind_has_its_own_check_id_and_detail(kind: AbortKind) -> None:
    assert kind.check_id
    assert kind.detail
    assert kind.check_id != "sut_alive" or kind is AbortKind.CRASHED
