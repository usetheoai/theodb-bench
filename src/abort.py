"""Why a run aborted, kept distinct from whether the system crashed.

Every abort invalidates its run -- a number from an aborted run is not a
measurement. What differs is what the report is allowed to say about it, and that
distinction is not cosmetic for a benchmark whose results are published: "the
system under test crashed" is a claim about the database, and making it when the
harness merely declined to measure, or when the harness's own time budget fired,
is a false claim about somebody's product.

Measured on 2026-08-17, in one session, all three reached the report as the same
sentence:

  * the knob gate refused a run because a requested search parameter was not in
    force on the server -- the harness working as designed;
  * `CREATE INDEX` on a million vectors was cancelled by the harness's own 60 s
    `statement_timeout`;
  * and nothing crashed: the container was `Up (healthy)`, its log held no PANIC
    and no FATAL.

Classification is by exception identity rather than by message text where
possible, and the unrecognised case is reported as a crash. That direction is
deliberate: mistaking a real crash for a refusal would hide a database defect
behind a harness message, which is the same error this module exists to prevent,
pointed the other way.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from theodb_bench.errors import AdapterError, BenchError, SystemUnavailableError


#: Driver exception names, checked in this order, mapped to what they mean.
#:
#: The order is load-bearing and verified against the installed driver:
#: `psycopg.errors.QueryCanceled` **is** a `psycopg.OperationalError`, so a
#: classifier that tested the connection-lost family first would call every
#: cancelled statement a crash -- precisely the misattribution this module exists
#: to remove. Cancellation is therefore checked before connection loss, and a test
#: pins that with the real exception classes rather than with stand-ins.
#:
#: Matched against the driver's own classes when it is importable, so subclasses
#: are covered; by qualified name otherwise, because the driver is an optional
#: dependency and the classifier has to work in a run that never had it.
def _driver_families() -> tuple[tuple[tuple[type[BaseException], ...], AbortKind], ...]:
    """Driver exception families, in classification order.

    Empty when the driver is absent, which leaves the name-based pass and the
    typed-error pass to do the work.
    """
    try:
        import psycopg
        import psycopg.errors as pg_errors
    except ImportError:  # pragma: no cover - exercised only without the extra
        return ()
    cancelled: tuple[type[BaseException], ...] = (pg_errors.QueryCanceled,)
    lost: tuple[type[BaseException], ...] = (
        pg_errors.AdminShutdown,
        pg_errors.CrashShutdown,
        pg_errors.ConnectionException,
        psycopg.OperationalError,
        psycopg.InterfaceError,
    )
    return ((cancelled, AbortKind.BUDGET_EXCEEDED), (lost, AbortKind.CRASHED))


#: Qualified names, used when the driver cannot be imported. Every entry was
#: verified to exist in the installed psycopg: a name that resolves to nothing is
#: a fabricated reference, and `psycopg.errors.StatementTimeout` -- which an
#: earlier draft of this list carried -- does not exist.
_CANCELLED_NAMES: frozenset[str] = frozenset(
    {"psycopg.errors.QueryCanceled", "psycopg2.extensions.QueryCanceledError"}
)

_CONNECTION_LOST_NAMES: frozenset[str] = frozenset(
    {
        "psycopg.OperationalError",
        "psycopg.InterfaceError",
        "psycopg.errors.AdminShutdown",
        "psycopg.errors.CrashShutdown",
        "psycopg.errors.ConnectionException",
        "psycopg2.OperationalError",
        "psycopg2.InterfaceError",
    }
)


class AbortKind(Enum):
    """What ended a run, and what the report may say about it."""

    CRASHED = "crashed"
    REFUSED = "refused"
    BUDGET_EXCEEDED = "budget_exceeded"

    @property
    def invalidates(self) -> bool:
        """Every abort invalidates the run. No number survives one."""
        return True

    @property
    def check_id(self) -> str:
        return {
            AbortKind.CRASHED: "sut_alive",
            AbortKind.REFUSED: "run_not_refused",
            AbortKind.BUDGET_EXCEEDED: "within_time_budget",
        }[self]

    @property
    def detail(self) -> str:
        return {
            AbortKind.CRASHED: "system under test crashed or became unreachable during the run",
            AbortKind.REFUSED: (
                "the harness refused to measure: a precondition it checks was not met, so no "
                "number was taken. This is the harness working, not a fault of the system "
                "under test"
            ),
            AbortKind.BUDGET_EXCEEDED: (
                "a statement was cancelled by a time budget the harness itself set. Raise the "
                "budget for this phase or reduce the scale; the system under test did not fail"
            ),
        }[self]


def _qualified_name(exc: BaseException) -> str:
    module = type(exc).__module__
    return f"{module}.{type(exc).__name__}" if module else type(exc).__name__


def classify_abort(exc: BaseException) -> AbortKind:
    """Classify the exception that ended a run.

    Order matters. `SystemUnavailableError` is checked before `AdapterError`
    because it is one -- an unreachable server is a crash, not a refusal, even
    though both arrive as the same base type.
    """
    for family, kind in _driver_families():
        if isinstance(exc, family):
            return kind

    name = _qualified_name(exc)
    if name in _CANCELLED_NAMES:
        return AbortKind.BUDGET_EXCEEDED
    if name in _CONNECTION_LOST_NAMES:
        return AbortKind.CRASHED
    if isinstance(exc, SystemUnavailableError):
        return AbortKind.CRASHED
    if isinstance(exc, (AdapterError, BenchError)):
        return AbortKind.REFUSED
    # Unrecognised: report the worst case. See the module docstring.
    return AbortKind.CRASHED


def describe_unavailability(adapter: Any, run_started_at: datetime) -> str:
    """Say whether the system went down and came back, or was merely unreachable.

    Both arrive as `SystemUnavailableError` and both are `CRASHED`, but they are
    different findings: one is a defect in the system under test, the other is
    usually the path to it. Measured 2026-08-17 — a 20 000 000-vector index build
    was OOM-killed and PostgreSQL recovered in 2.98 s, and separating the two took
    reading `docker logs` and `dmesg` by hand, outside the bundle and unavailable
    to anyone reading it later.

    `pg_postmaster_start_time()` settles it in one query with no privileged
    access: a start time later than the run's start means the server went down
    and came back.

    Deliberately not attempted: reading the server's own log, or the kernel's.
    Both need access the harness has no claim to, and a benchmark that silently
    requires root on the database host is one most people cannot run.

    This describes a crash; it never reclassifies one. The abort stays `CRASHED`
    and the run stays invalid.
    """
    probe = getattr(adapter, "postmaster_start_time", None)
    if probe is None:
        # Only the PostgreSQL family can answer this. Silence beats failing a
        # run for a diagnostic it was never able to produce.
        return ""

    try:
        started = probe()
    except Exception as exc:  # the system is already known to be unreachable
        return (
            f"could not ask the system when it last started, so whether it "
            f"restarted is unknown: {type(exc).__name__}"
        )

    if started is None:
        return "could not ask the system when it last started; whether it restarted is unknown"

    if started > run_started_at:
        elapsed = started - run_started_at
        return (
            f"the system restarted {int(elapsed.total_seconds() // 60)} min into the run "
            f"(postmaster start {started.isoformat()}, run start {run_started_at.isoformat()}). "
            f"It went down and came back, which is a finding about the system under test."
        )

    return (
        f"the system did not restart (postmaster up since {started.isoformat()}, before the "
        f"run began). The connection to it failed while it stayed up, which usually points at "
        f"the environment rather than the system under test."
    )
