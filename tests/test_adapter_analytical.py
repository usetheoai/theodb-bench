"""The analytical path of the PostgreSQL-family adapters.

Everything asserted here was measured against real servers on an ephemeral
droplet (138.197.22.192, 2026-08-17) before it was written down, because the
central fact of this surface is that the obvious residency proof does not prove
residency.

AlloyDB Omni's columnar engine is a *cache populated by policy*, and it has four
distinguishable states:

  1. `google_columnar_engine.enabled = off` -- the default. Querying
     `g_columnar_columns` errors outright.
  2. enabled, store never populated -- `g_columnar_columns` = 0, plan = Seq Scan.
  3. enabled, `google_columnar_engine_add()` called, container `/dev/shm` at
     Docker's 64 MB default -- `g_columnar_columns` = **4** while
     `g_columnar_engine_summary` reports **Memory Used = 0 MB**, and the plan is
     still a Seq Scan. The refresh fails with
     `could not resize shared memory segment ... No space left on device`.
  4. enabled, populated, `--shm-size=4g` -- Memory Used = 42 MB and the plan
     carries `Parallel Custom Scan (columnar scan)`.

State 3 is why this file exists. `g_columnar_columns` reports **registration**,
not residency, so a gate built on it passes while the store is empty and the run
measures a sequential scan under the columnar engine's name. It is the same shape
as `current_setting` versus `pg_settings`: the obvious instrument reports the
request, not the effect.

TheoDB's columnar is a *table access method*, so `pg_class.relam` proves
residency structurally -- there is no "enabled but empty" state to detect.
"""

from __future__ import annotations

from typing import Any

import pytest
from theodb_bench.adapters.alloydb import AlloyDBOmniAdapter
from theodb_bench.adapters.base import AnalyticalQuery, AnalyticalTable
from theodb_bench.adapters.postgres import PostgresAdapter, TheoDBAdapter
from theodb_bench.errors import AdapterError, UnsupportedCapabilityError

COLUMNS = ("id", "amount", "category", "quantity")
ROWS: list[tuple[Any, ...]] = [
    (0, 10.5, "a", 3),
    (1, -4.25, "b", 7),
    (2, 30.0, "a", 12),
]


class _AnalyticalStub:
    """A server that answers the catalog and plan queries a gate reads."""

    def __init__(
        self,
        *,
        relam: str = "heap",
        columnar_columns: int = 0,
        memory_used_mb: int = 0,
        engine_enabled: str = "on",
        plan: str = "Seq Scan on bench_analytical_columnar",
    ) -> None:
        self.executed: list[str] = []
        self.relam = relam
        self.columnar_columns = columnar_columns
        self.memory_used_mb = memory_used_mb
        self.engine_enabled = engine_enabled
        self.plan = plan
        self.rows_by_query: dict[str, tuple[tuple[Any, ...], ...]] = {}
        #: GUC name -> (setting, source), as `pg_settings` answers it.
        self.settings: dict[str, tuple[str, str]] = {}

    def execute(self, sql: str, parameters: tuple[object, ...] | None = None) -> None:
        self.executed.append(sql)

    def cursor(self) -> Any:
        return _CursorStub(self)

    def fetch_one(self, sql: str, parameters: tuple[object, ...] | None = None):
        # The gate binds the setting name and the relation name as parameters, so
        # a stub that only inspected the SQL would answer the wrong question.
        first = str(parameters[0]) if parameters else ""
        if "pg_settings" in sql and first == "google_columnar_engine.enabled":
            return (self.engine_enabled, "postmaster")
        if "pg_settings" in sql and first in self.settings:
            return self.settings[first]
        if "pg_settings" in sql and first.startswith("theodb."):
            return None
        if "pg_am" in sql and "relam" in sql:
            return (self.relam,)
        if "g_columnar_columns" in sql:
            return (self.columnar_columns,)
        if "g_columnar_engine_summary" in sql:
            return (self.memory_used_mb,)
        if "EXPLAIN" in sql:
            return (self.plan,)
        if "count(*)" in sql:
            return (len(ROWS),)
        return None

    def fetch_all(self, sql: str, parameters: tuple[object, ...] | None = None):
        if "EXPLAIN" in sql:
            return [(self.plan,)]
        for query_id, rows in self.rows_by_query.items():
            if query_id in sql:
                return list(rows)
        return [(len(ROWS),)]


class _CursorStub:
    """The context-manager cursor `load_analytical` batches inserts through."""

    def __init__(self, server: _AnalyticalStub) -> None:
        self._server = server

    def __enter__(self) -> _CursorStub:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def executemany(self, sql: str, batch: list[tuple[Any, ...]]) -> None:
        self._server.executed.append(sql)


def _wire(adapter: Any, server: _AnalyticalStub) -> Any:
    adapter._execute = server.execute
    adapter._fetch_one = server.fetch_one
    adapter._fetch_all = server.fetch_all
    adapter._cursor = server.cursor
    return adapter


# ------------------------------------------------------------------ heap path


def test_the_row_path_creates_a_heap_table_and_loads_it() -> None:
    server = _AnalyticalStub()
    adapter = _wire(PostgresAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_row", columns=COLUMNS, path="row")

    outcome = adapter.load_analytical(table, ROWS)

    statements = " | ".join(server.executed)
    assert "CREATE TABLE" in statements
    assert "USING" not in statements  # heap is the default; naming it adds nothing
    assert outcome.rows_loaded == len(ROWS)
    assert outcome.complete


def test_upstream_postgres_refuses_the_columnar_path() -> None:
    """It has no columnar surface, and pretending otherwise would measure heap."""
    server = _AnalyticalStub()
    adapter = _wire(PostgresAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    with pytest.raises(UnsupportedCapabilityError):
        adapter.load_analytical(table, ROWS)


@pytest.mark.parametrize(
    "query_id", ["total_rows", "sum_amount", "group_by_category", "filtered_sum"]
)
def test_every_declared_query_has_sql(query_id: str) -> None:
    server = _AnalyticalStub()
    adapter = _wire(PostgresAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="row")

    result = adapter.execute_analytical(table, AnalyticalQuery(id=query_id, description=""))

    assert result.wall_seconds >= 0.0


def test_an_unknown_query_is_refused_not_silently_answered() -> None:
    server = _AnalyticalStub()
    adapter = _wire(PostgresAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="row")

    with pytest.raises(AdapterError, match="unknown analytical query"):
        adapter.execute_analytical(table, AnalyticalQuery(id="invented", description=""))


# ------------------------------------------------------- TheoDB: columnar TAM


def test_theodb_creates_the_columnar_table_with_its_access_method() -> None:
    server = _AnalyticalStub(relam="theodb_columnar")
    server.settings["theodb.enable_columnar_agg"] = ("on", "session")
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    adapter.load_analytical(table, ROWS)

    statements = " | ".join(server.executed)
    assert "USING theodb_columnar" in statements


def test_theodb_refuses_a_columnar_table_that_is_really_heap() -> None:
    """`pg_class.relam` is the whole proof, so a heap table must not pass as columnar."""
    server = _AnalyticalStub(relam="heap")
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError, match="heap"):
        adapter.assert_analytical_path(table)


def test_theodb_accepts_a_genuinely_columnar_table() -> None:
    server = _AnalyticalStub(
        relam="theodb_columnar",
        plan="Custom Scan (theodb_columnar_agg) on bench_analytical_columnar",
    )
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    adapter.assert_analytical_path(table)  # must not raise


# ------------------------------------------------ Omni: the four measured states


def test_omni_refuses_when_the_engine_is_off() -> None:
    """State 1, the default. `context=postmaster`: only a restart turns it on."""
    server = _AnalyticalStub(engine_enabled="off")
    adapter = _wire(AlloyDBOmniAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError, match="restart"):
        adapter.assert_analytical_path(table)


def test_omni_refuses_when_nothing_is_registered() -> None:
    """State 2: enabled, store never populated."""
    server = _AnalyticalStub(engine_enabled="on", columnar_columns=0, memory_used_mb=0)
    adapter = _wire(AlloyDBOmniAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError, match="no column of"):
        adapter.assert_analytical_path(table)


def test_omni_refuses_registered_columns_with_an_empty_store() -> None:
    """State 3, and the reason this gate is not built on g_columnar_columns.

    Measured: the view reported 4 columns while the engine summary reported
    Memory Used = 0 MB, and the plan was a sequential scan. A gate that stopped
    at the view would have published heap timings as columnar.
    """
    server = _AnalyticalStub(engine_enabled="on", columnar_columns=4, memory_used_mb=0)
    adapter = _wire(AlloyDBOmniAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError, match=r"registered .* not loaded|shared memory"):
        adapter.assert_analytical_path(table)


def test_omni_refuses_a_plan_that_does_not_use_the_columnar_scan() -> None:
    """Residency is necessary and not sufficient: measured resident at 50 000 rows
    with the planner still choosing a sequential scan."""
    server = _AnalyticalStub(
        engine_enabled="on",
        columnar_columns=4,
        memory_used_mb=42,
        plan="Parallel Seq Scan on t",
    )
    adapter = _wire(AlloyDBOmniAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError, match="did not use the columnar"):
        adapter.assert_analytical_path(table)


def test_omni_accepts_a_loaded_store_that_the_plan_actually_uses() -> None:
    """State 4, the only one that may be measured."""
    server = _AnalyticalStub(
        engine_enabled="on",
        columnar_columns=4,
        memory_used_mb=42,
        plan="Parallel Custom Scan (columnar scan) on t",
    )
    adapter = _wire(AlloyDBOmniAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    adapter.assert_analytical_path(table)  # must not raise


def test_omni_registers_the_table_with_the_engine_after_loading() -> None:
    server = _AnalyticalStub(engine_enabled="on", columnar_columns=4, memory_used_mb=42)
    adapter = _wire(AlloyDBOmniAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    adapter.load_analytical(table, ROWS)

    statements = " | ".join(server.executed)
    assert "google_columnar_engine_add" in statements


def test_omni_declares_the_columnar_capability() -> None:
    assert AlloyDBOmniAdapter().capabilities()["columnar"] is True


def test_theodb_declares_the_columnar_capability() -> None:
    assert TheoDBAdapter().capabilities()["columnar"] is True


# ------------------------ the columnar pushdown, and why residency is not enough
#
# Measured on the built image, same table (1M rows), same query, same session:
#
#   theodb.enable_columnar_agg = off (the DEFAULT)  -> Seq Scan          1407 ms
#   theodb.enable_columnar_agg = on                 -> Custom Scan
#                                                      (theodb_columnar_agg)  108 ms
#
# A 13x difference decided by a GUC that ships off. A gate that proved only
# `pg_class.relam` would have published the 1407 ms as "our columnar", which is
# columnar storage without its pushdown -- a path the project already knows is
# slower than heap. It is the same shape as the competitor's
# `scann.enable_ah_quantizer = off`.


def test_theodb_enables_the_columnar_aggregate_pushdown() -> None:
    server = _AnalyticalStub(relam="theodb_columnar")
    server.settings["theodb.enable_columnar_agg"] = ("on", "session")
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    adapter.load_analytical(table, ROWS)

    statements = " | ".join(server.executed)
    assert "SET theodb.enable_columnar_agg = on" in statements


def test_theodb_refuses_when_the_pushdown_guc_did_not_take() -> None:
    """The B-060 gate, on the analytical axis: a SET that did not apply is a refusal."""
    server = _AnalyticalStub(relam="theodb_columnar")  # no setting registered
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError):
        adapter.load_analytical(table, ROWS)


def test_theodb_refuses_a_plan_that_never_reaches_the_columnar_scan() -> None:
    """Residency is necessary and not sufficient -- 13x not sufficient."""
    server = _AnalyticalStub(relam="theodb_columnar", plan="Seq Scan on bench_analytical_columnar")
    server.settings["theodb.enable_columnar_agg"] = ("on", "session")
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    with pytest.raises(AdapterError, match="did not use"):
        adapter.assert_analytical_path(table)


def test_theodb_accepts_a_plan_that_uses_the_columnar_pushdown() -> None:
    server = _AnalyticalStub(
        relam="theodb_columnar",
        plan="Custom Scan (theodb_columnar_agg) on bench_analytical_columnar",
    )
    server.settings["theodb.enable_columnar_agg"] = ("on", "session")
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="bench_analytical_columnar", columns=COLUMNS, path="columnar")

    adapter.assert_analytical_path(table)  # must not raise


def test_the_plan_proof_is_per_query_not_per_table() -> None:
    """Pushdown coverage varies by query shape, so proving it once is not proving it.

    Measured at one million rows with the pushdown on: `sum(amount)` plans as
    `Custom Scan (theodb_columnar_agg)`, while `GROUP BY category` falls back to
    Seq Scan -> external-merge Sort (25 456 kB spilled) -> GroupAggregate and runs
    14x slower than heap. A gate that probed `filtered_sum` and generalised would
    have called the grouped query pushed down.
    """

    class _PerQueryStub(_AnalyticalStub):
        def fetch_one(self, sql: str, parameters: tuple[object, ...] | None = None):
            if "EXPLAIN" in sql and "GROUP BY" in sql:
                return ("GroupAggregate -> Sort -> Seq Scan on t",)
            if "EXPLAIN" in sql:
                return ("Custom Scan (theodb_columnar_agg) on t",)
            return super().fetch_one(sql, parameters)

    server = _PerQueryStub(relam="theodb_columnar")
    server.settings["theodb.enable_columnar_agg"] = ("on", "session")
    adapter = _wire(TheoDBAdapter(), server)
    table = AnalyticalTable(name="t", columns=COLUMNS, path="columnar")

    # the scalar aggregate pushes down
    adapter.assert_analytical_path(table, AnalyticalQuery(id="sum_amount", description=""))

    # the grouped one does not, and must be refused rather than measured
    with pytest.raises(AdapterError, match="group_by_category"):
        adapter.assert_analytical_path(
            table, AnalyticalQuery(id="group_by_category", description="")
        )
