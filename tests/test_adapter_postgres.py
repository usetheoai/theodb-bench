"""PostgreSQL adapters: the parts that can be proven without a server.

SQL construction, index sizing, opclass resolution and identifier handling are
pure functions of the configuration, and they are exactly where the fairness
invariants live. Behaviour that needs a live server is marked `integration`.
"""

from __future__ import annotations

import numpy as np
import pytest
from theodb_bench.adapters.base import IndexSpec, KnnQuery, SystemAdapter, VectorTableSpec
from theodb_bench.adapters.postgres import (
    PgvectorAdapter,
    PostgresAdapter,
    TheoDBAdapter,
    clamp_probes,
    ivfflat_lists,
)
from theodb_bench.errors import AdapterError, UnsupportedCapabilityError
from theodb_bench.schemas import validate

SPEC = VectorTableSpec(table="bench_vectors", dimension=8, metric="l2")


# -------------------------------------------------------------- ivfflat sizing


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(1_000, 1), (50_000, 50), (1_000_000, 1000), (4_000_000, 2000)],
)
def test_ivfflat_lists_follows_the_documented_rule(rows: int, expected: int) -> None:
    assert ivfflat_lists(rows) == expected


def test_ivfflat_lists_never_returns_zero() -> None:
    # rows // 1000 is 0 below a thousand rows, and lists=0 is invalid DDL.
    assert ivfflat_lists(500) == 1


def test_ivfflat_lists_refuses_an_empty_table() -> None:
    with pytest.raises(AdapterError, match="empty table"):
        ivfflat_lists(0)


def test_lists_derived_from_the_real_count_differs_from_a_default() -> None:
    # The defect this prevents: sizing a million-row index from a 5000-row
    # default builds lists=5 and calls the result a comparison.
    assert ivfflat_lists(1_000_000) != ivfflat_lists(5_000)


@pytest.mark.parametrize(
    ("probes", "lists", "expected"),
    [(1, 10, 1), (10, 10, 10), (100, 10, 10), (0, 10, 1)],
)
def test_probes_are_clamped_to_lists(probes: int, lists: int, expected: int) -> None:
    # probes > lists is a no-op in pgvector; reporting it under its own label
    # would put a duplicate point in the table under a name that lies.
    assert clamp_probes(probes, lists) == expected


# -------------------------------------------------------------------- opclass


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_pgvector_resolves_an_opclass_for_every_supported_metric(metric: str) -> None:
    assert PgvectorAdapter().opclass("hnsw", metric).startswith("vector_")


def test_an_unavailable_opclass_is_unsupported_not_invented() -> None:
    # Emitting DDL for an opclass that does not exist turns a missing
    # capability into a failed run.
    with pytest.raises(UnsupportedCapabilityError, match="no hnsw operator class"):
        PgvectorAdapter().opclass("hnsw", "manhattan")


def test_an_unknown_access_method_has_no_opclasses() -> None:
    with pytest.raises(UnsupportedCapabilityError):
        PgvectorAdapter().opclass("cuckoo", "l2")


# ------------------------------------------------------------------- index ddl


def test_hnsw_ddl_names_the_opclass_and_parameters() -> None:
    adapter = PgvectorAdapter()
    adapter._row_count = 10_000
    name, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="hnsw", parameters={"m": 16}))
    assert name == "bench_vectors_hnsw_l2_idx"
    assert "USING hnsw" in ddl
    assert "vector_l2_ops" in ddl
    assert "m = 16" in ddl


def test_ivfflat_ddl_injects_lists_from_the_real_row_count() -> None:
    adapter = PgvectorAdapter()
    adapter._row_count = 250_000
    _, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="ivfflat"))
    assert f"lists = {ivfflat_lists(250_000)}" in ddl


def test_an_explicit_lists_value_is_respected() -> None:
    adapter = PgvectorAdapter()
    adapter._row_count = 250_000
    _, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="ivfflat", parameters={"lists": 7}))
    assert "lists = 7" in ddl


def test_ddl_uses_the_metric_declared_by_the_table() -> None:
    adapter = PgvectorAdapter()
    adapter._row_count = 1_000
    cosine_spec = VectorTableSpec(table="bench_vectors", dimension=8, metric="cosine")
    _, ddl = adapter.index_ddl(cosine_spec, IndexSpec(kind="hnsw"))
    assert "vector_cosine_ops" in ddl


# ----------------------------------------------------------------------- SQL


def test_query_orders_by_distance_then_id() -> None:
    # Without the id, ties at the top-k boundary resolve by physical row order,
    # which changes between runs and makes recall irreproducible.
    adapter = PgvectorAdapter()
    sql = adapter._query_sql(
        KnnQuery(table="bench_vectors", vector=np.zeros(8, dtype=np.float32), k=10)
    )
    assert "ORDER BY" in sql
    assert sql.strip().endswith("LIMIT 10")
    assert ", id LIMIT" in sql


def test_upstream_postgres_computes_the_distance_in_sql() -> None:
    # Upstream PostgreSQL has no distance operator; the arithmetic must be the
    # same one the oracle performs.
    sql = PostgresAdapter()._query_sql(
        KnnQuery(table="bench_vectors", vector=np.zeros(8, dtype=np.float32), k=5)
    )
    assert "unnest" in sql
    assert "ORDER BY distance, id" in sql


@pytest.mark.parametrize(("metric", "operator"), [("l2", "<->"), ("ip", "<#>"), ("cosine", "<=>")])
def test_pgvector_uses_the_right_distance_operator(metric: str, operator: str) -> None:
    assert operator in PgvectorAdapter().distance_expression(metric, '"embedding"')


def test_an_unknown_metric_is_refused_by_both_adapters() -> None:
    with pytest.raises(AdapterError, match="unknown metric"):
        PgvectorAdapter().distance_expression("manhattan", '"embedding"')
    with pytest.raises(AdapterError, match="unknown metric"):
        PostgresAdapter().distance_expression("manhattan", '"embedding"')


@pytest.mark.parametrize(
    "table", ["bench; DROP TABLE users", "a b", "", 'x"y', "tab\tname", "../etc"]
)
def test_identifiers_that_are_not_identifiers_are_refused(table: str) -> None:
    # A benchmark definition is data, and data does not get to write SQL.
    adapter = PgvectorAdapter()
    with pytest.raises(AdapterError, match="invalid SQL identifier"):
        adapter.index_ddl(
            VectorTableSpec(table=table, dimension=4, metric="l2"), IndexSpec(kind="hnsw")
        )


def test_column_types_differ_between_upstream_and_pgvector() -> None:
    assert PostgresAdapter().column_type(128) == "real[]"
    assert PgvectorAdapter().column_type(128) == "vector(128)"


# --------------------------------------------------------------- capabilities


def test_upstream_postgres_declares_only_exact_search() -> None:
    adapter = PostgresAdapter()
    assert adapter.supports("vector_exact")
    assert not adapter.supports("vector_hnsw")
    assert not adapter.supports("vector_ivfflat")


def test_upstream_postgres_refuses_to_pretend_it_has_an_index() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="vector_hnsw"):
        PostgresAdapter().build_index(SPEC, IndexSpec(kind="hnsw"))


def test_pgvector_declares_both_index_families() -> None:
    adapter = PgvectorAdapter()
    assert adapter.supports("vector_hnsw")
    assert adapter.supports("vector_ivfflat")


def test_theodb_declares_only_the_vector_surface_it_can_exercise() -> None:
    # TheoDB the database has hybrid, columnar, Parquet, graph and a
    # vectorizer. This adapter reaches none of them, and declaring them would
    # put a false claim into every system.json and into `theodb-bench list`.
    adapter = TheoDBAdapter()
    assert adapter.supports("vector_hnsw")
    for capability in ("hybrid", "lexical", "columnar", "parquet", "graph", "vectorizer"):
        assert not adapter.supports(capability), capability


# The surfaces beyond vector need these adapter methods. A capability may only
# be declared once the corresponding method is implemented.
_CAPABILITY_METHODS: dict[str, tuple[str, ...]] = {
    "lexical": ("load_documents", "execute_lexical"),
    "hybrid": ("load_documents", "execute_hybrid"),
    "rerank": ("execute_rerank",),
    "graph": ("load_graph", "traverse", "graph_stats"),
    "columnar": ("load_analytical", "execute_analytical"),
    "parquet": ("load_analytical", "execute_analytical"),
    "vectorizer": ("insert_document", "is_fresh", "queue_depth", "vectorizer_stats"),
}


@pytest.mark.parametrize("adapter_cls", [PostgresAdapter, PgvectorAdapter, TheoDBAdapter])
def test_no_adapter_declares_a_capability_it_cannot_exercise(adapter_cls: type) -> None:
    """Structural guard against the defect this file previously encoded.

    A capability is a statement about the adapter's own code path. Declaring
    one whose lifecycle methods fall through to the base class puts a false
    claim into system.json, where a reader has no way to check it.
    """
    adapter = adapter_cls()
    base_methods = {
        name: getattr(SystemAdapter, name, None)
        for names in _CAPABILITY_METHODS.values()
        for name in names
    }
    for capability, methods in _CAPABILITY_METHODS.items():
        if not adapter.supports(capability):
            continue
        for name in methods:
            assert getattr(type(adapter), name, None) is not base_methods[name], (
                f"{adapter.system_id} declares {capability!r} but inherits {name}() "
                "from the base class, which only raises"
            )


def test_theodb_uses_its_own_extension() -> None:
    assert TheoDBAdapter().extension == "theodb_rs"
    assert PgvectorAdapter().extension == "vector"


@pytest.mark.parametrize("adapter_cls", [PostgresAdapter, PgvectorAdapter, TheoDBAdapter])
def test_capabilities_are_declared_over_the_closed_vocabulary(adapter_cls: type) -> None:
    from theodb_bench.adapters.base import CAPABILITIES

    adapter = adapter_cls()
    declared = {name: adapter.supports(name) for name in CAPABILITIES}
    assert set(declared) == set(CAPABILITIES)


def test_system_payload_shape_is_schema_valid_without_a_server() -> None:
    # export_config needs a connection, so this checks the part that does not:
    # the identity and capability half of system.json.
    adapter = PgvectorAdapter()
    payload = {
        "schema_version": 1,
        "system": adapter.system_id,
        "capabilities": adapter.capabilities(),
    }
    validate("system", payload)


# ------------------------------------------------- search-parameter gate (B-060)
#
# The harness already refuses to report a number when the planner ignored the index
# (`assert_index_used`, postgres.py). It did not refuse when the *knob* was ignored: the
# SET was issued and nothing read the value back.
#
# That gap is not theoretical, and the mechanism was measured on PostgreSQL 18:
#
#     SET nao.existe = 999;                          -> SET   (succeeds)
#     SELECT current_setting('nao.existe', true);     -> 999   (echoes what we wrote)
#     SELECT count(*) FROM pg_settings WHERE name='nao.existe';  -> 0
#
# An unregistered namespaced GUC is accepted as a placeholder. `current_setting` therefore
# cannot detect it — it hands back our own value. `pg_settings` can, because it lists only
# *registered* GUCs.
#
# Two independent measurements show the class bites: TheoDB's own B-034 (`SET hnsw.ef_search`
# accepted with no effect) and the 2026-08-15 AlloyDB evaluation, where
# `scann.num_leaves_to_search` silently does nothing without `LOAD 'alloydb_scann'` — the
# evaluator asked for a deep search, got recall 0.15, and lost a 10M-vector run.


class _ServerStub:
    """A server that answers `pg_settings` however the test wants.

    Mirrors the shape the gate reads: (setting, source) for a GUC name, or None when the
    GUC is not registered — which is what a server whose extension library never loaded
    looks like.
    """

    def __init__(self, settings: dict[str, tuple[str, str]]) -> None:
        self._settings = settings
        self.executed: list[str] = []

    def execute(self, sql: str, parameters: tuple[object, ...] | None = None) -> None:
        self.executed.append(sql)

    def fetch_one(self, sql: str, parameters: tuple[object, ...] | None = None):
        name = parameters[0] if parameters else None
        entry = self._settings.get(str(name))
        return None if entry is None else entry


def _adapter_with(server: _ServerStub) -> PgvectorAdapter:
    adapter = PgvectorAdapter()
    adapter._row_count = 10_000
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]
    return adapter


def test_gate_accepts_a_knob_the_server_actually_applied() -> None:
    server = _ServerStub({"hnsw.ef_search": ("200", "session")})
    adapter = _adapter_with(server)

    adapter.set_search_parameters({"ef_search": 200})

    assert adapter.effective_search_parameters() == {"hnsw.ef_search": "200"}


def test_gate_refuses_an_adapter_that_accepts_the_knob_and_ignores_it() -> None:
    """The placeholder case: the SET succeeds, the GUC is not registered, nothing applied.

    This is the exact mechanism measured on PostgreSQL 18 and the one that cost the
    independent AlloyDB evaluation a ten-million-vector run.
    """
    server = _ServerStub({})  # nothing registered: the library never loaded
    adapter = _adapter_with(server)

    with pytest.raises(AdapterError) as excinfo:
        adapter.set_search_parameters({"ef_search": 200})

    message = str(excinfo.value)
    assert "hnsw.ef_search" in message
    assert "requested=" in message
    assert "effective=" in message


def test_gate_refuses_a_registered_knob_the_set_did_not_move() -> None:
    """Registered but still at its default means the SET did not take effect."""
    server = _ServerStub({"hnsw.ef_search": ("40", "default")})
    adapter = _adapter_with(server)

    with pytest.raises(AdapterError, match="effective="):
        adapter.set_search_parameters({"ef_search": 200})


def test_gate_accepts_the_documented_clamp() -> None:
    """`probes` is clamped to the list count, so the effective value legitimately differs
    from what the caller asked for. The gate compares against what was SENT, not the raw
    request — otherwise the harness's own sizing rule would trip it."""
    lists = ivfflat_lists(10_000)
    sent = clamp_probes(10_000, lists)
    server = _ServerStub({"ivfflat.probes": (str(sent), "session")})
    adapter = _adapter_with(server)

    adapter.set_search_parameters({"probes": 10_000})

    assert adapter.effective_search_parameters() == {"ivfflat.probes": str(sent)}


def test_gate_distinguishes_could_not_verify_from_verified_and_divergent() -> None:
    """A read that fails is not a divergence. Collapsing the two would report a
    configuration defect where there was an unavailable server — the distinction
    `cycle-acceptance` protects with NOT_VALIDATED."""
    class _Unreadable(_ServerStub):
        def fetch_one(self, sql: str, parameters: tuple[object, ...] | None = None):
            raise RuntimeError("connection reset")

    adapter = _adapter_with(_Unreadable({}))
    with pytest.raises(AdapterError, match="could not verify"):
        adapter.set_search_parameters({"ef_search": 200})


@pytest.mark.parametrize("name", ["postgres", "pgvector", "theodb", "fake"])
def test_every_adapter_reports_effective_search_parameters(name: str) -> None:
    """A new engine cannot be added without answering what is in force.

    Without this, the contract holds for three adapters out of four and the fourth breaks
    it in silence — and `fake` is the one the runner's own tests exercise most.
    """
    from theodb_bench.registry import get_adapter

    entry = get_adapter(name)
    assert hasattr(entry.factory, "effective_search_parameters") or hasattr(
        entry.factory(), "effective_search_parameters"
    ), f"{name} does not report its effective search parameters"
