"""PostgreSQL adapters: the parts that can be proven without a server.

SQL construction, index sizing, opclass resolution and identifier handling are
pure functions of the configuration, and they are exactly where the fairness
invariants live. Behaviour that needs a live server is marked `integration`.
"""

from __future__ import annotations

import struct
from typing import ClassVar

import numpy as np
import pytest
from theodb_bench.adapters.base import (
    CAPABILITIES,
    AnalyticalTable,
    IndexSpec,
    KnnQuery,
    SystemAdapter,
    VectorTableSpec,
)
from theodb_bench.adapters.postgres import (
    PgvectorAdapter,
    PostgresAdapter,
    PostgresConfig,
    TheoDBAdapter,
    clamp_probes,
    ivfflat_lists,
)
from theodb_bench.copy_binary import BINARY_HEADER, BINARY_TRAILER
from theodb_bench.errors import AdapterError, ErrorContext, Phase, UnsupportedCapabilityError
from theodb_bench.registry import ADAPTERS, get_adapter
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


def test_theodb_declares_only_the_surface_it_can_exercise() -> None:
    """TheoDB the database has hybrid, columnar, Parquet, graph and a vectorizer.

    `columnar` left this list when the analytical surface arrived: the adapter
    creates the table `USING theodb_columnar` and proves residency from
    `pg_class.relam`, so the claim is now checkable. The rest stay out — a
    capability is a statement about this code path, and declaring one whose
    lifecycle methods fall through to the base puts a false claim into every
    `system.json`.
    """
    adapter = TheoDBAdapter()
    assert adapter.supports("vector_hnsw")
    assert adapter.supports("columnar")
    # Reached as of the pillar work: `write_parquet`/`read_parquet` for one,
    # `bm25_build`/`bm25_search` for the other.
    assert adapter.supports("parquet")
    assert adapter.supports("lexical")
    # `hybrid` needs both legs measured together and is not wired yet; `rerank`,
    # `vectorizer` and `ai_sql` each reach an external model, and without an
    # endpoint there is nothing to measure.
    #  folds the CSR,  walks it.
    assert adapter.supports("graph")
    #  fuses both legs; the quantizer reloptions are real
    # and the pg-scann suite builds with pq_subspaces=64.
    assert adapter.supports("hybrid")
    assert adapter.supports("vector_quantized")
    for capability in ("vectorizer", "rerank", "ai_sql"):
        assert not adapter.supports(capability), capability


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_a_declared_analytical_capability_has_a_path_to_run_it(name: str) -> None:
    """`columnar` and `parquet` share their lifecycle methods, so the structural
    guard cannot tell them apart. An adapter that declared `parquet` while its
    `ANALYTICAL_PATHS` had no such entry would pass that guard and refuse at
    runtime -- which is safe, but late.
    """
    adapter = ADAPTERS[name].factory()
    paths = getattr(type(adapter), "ANALYTICAL_PATHS", None)
    if paths is None:
        pytest.skip(f"{name} is not a PostgreSQL-family adapter")

    for capability, path in (("columnar", "columnar"), ("parquet", "parquet")):
        if adapter.supports(capability):
            assert path in paths, f"{name} declares {capability} with no {path!r} path"


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

    def fetch_one(
        self, sql: str, parameters: tuple[object, ...] | None = None
    ) -> tuple[object, ...] | None:
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
        def fetch_one(
            self, sql: str, parameters: tuple[object, ...] | None = None
        ) -> tuple[object, ...] | None:
            raise RuntimeError("connection reset")

    adapter = _adapter_with(_Unreadable({}))
    with pytest.raises(AdapterError, match="could not verify"):
        adapter.set_search_parameters({"ef_search": 200})


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_adapter_reports_effective_search_parameters(name: str) -> None:
    """A new engine cannot be added without answering what is in force.

    Without this, the contract holds for three adapters out of four and the fourth breaks
    it in silence — and `fake` is the one the runner's own tests exercise most.

    Parametrized off `ADAPTERS` rather than a written-out list. The list version was
    measured passing while `alloydbomni` was already registered and uncovered: a test
    that enumerates what it claims to cover universally excludes every adapter added
    after it was written, and reports green for doing so.
    """
    entry = get_adapter(name)
    assert hasattr(entry.factory, "effective_search_parameters") or hasattr(
        entry.factory(), "effective_search_parameters"
    ), f"{name} does not report its effective search parameters"


def test_alloydbomni_is_registered() -> None:
    entry = get_adapter("alloydbomni")

    assert entry.requires == ("psycopg",)
    assert "query layer" in entry.description
    assert "PostgreSQL 17" in entry.description  # the measured version, not the tag


# ------------------------------------------------ reloption rendering by type
#
# Measured on google/alloydbomni:latest (droplet 138.197.22.192, 2026-08-17):
#   CREATE INDEX ... USING scann (emb cosine) WITH (num_leaves=10, quantizer='sq8')
#   -> CREATE INDEX; pg_class.reloptions = {num_leaves=10, quantizer=sq8}
# `quantizer` is a string, and the renderer only knew how to write integers.


class _ScannLike(PgvectorAdapter):
    """An engine whose access method takes a string reloption, as scann does."""

    OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {
        "scann": {"l2": "l2", "ip": "dot_product", "cosine": "cosine"}
    }


def test_a_string_reloption_is_rendered_quoted() -> None:
    adapter = _ScannLike()
    adapter._row_count = 200

    _, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="scann", parameters={"quantizer": "sq8"}))

    assert "quantizer = 'sq8'" in ddl


def test_an_int_reloption_is_still_rendered_bare() -> None:
    """Regression: the existing integer path must not gain quotes."""
    adapter = PgvectorAdapter()
    adapter._row_count = 200

    _, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="hnsw", parameters={"m": 16}))

    assert "m = 16" in ddl


def test_an_unrenderable_reloption_raises_adapter_error_not_value_error() -> None:
    """A bad benchmark definition must fail with phase and system, not a bare ValueError."""
    adapter = PgvectorAdapter()
    adapter._row_count = 200

    with pytest.raises(AdapterError) as exc:
        adapter.index_ddl(SPEC, IndexSpec(kind="hnsw", parameters={"m": [1, 2]}))

    assert "m" in str(exc.value)


def test_a_string_reloption_cannot_inject_sql() -> None:
    """Benchmark definitions are data, and data does not get to write SQL."""
    adapter = _ScannLike()
    adapter._row_count = 200

    _, ddl = adapter.index_ddl(
        SPEC, IndexSpec(kind="scann", parameters={"quantizer": "sq8') OR '1'='1"})
    )

    assert "''" in ddl


def test_setting_opclasses_on_an_instance_does_not_change_the_lookup() -> None:
    """The table is read off the class on purpose.

    An adapter declares its convention by subclassing, so a stray instance
    attribute cannot silently redirect which operator class an index is built
    with.
    """
    adapter = PgvectorAdapter()
    adapter.OPCLASSES = {"scann": {"l2": "l2"}}  # type: ignore[misc]

    with pytest.raises(UnsupportedCapabilityError):
        adapter.opclass("scann", "l2")


# ------------------------------------------------------- per-adapter opclasses
#
# Measured on the same droplet: the scann access method names its operator
# classes `cosine` / `dot_product` / `l2` — none of which match pgvector's
# `vector_*_ops` convention. A shared module-level table cannot serve both.


def test_an_adapter_declares_its_own_opclasses() -> None:
    class _Stub(PgvectorAdapter):
        OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {
            "scann": {"cosine": "cosine", "ip": "dot_product", "l2": "l2"}
        }

    assert _Stub().opclass("scann", "cosine") == "cosine"
    assert _Stub().opclass("scann", "ip") == "dot_product"


def test_the_pgvector_convention_survives_the_move() -> None:
    assert PgvectorAdapter().opclass("hnsw", "cosine") == "vector_cosine_ops"


def test_an_unknown_metric_names_what_is_available() -> None:
    class _Stub(PgvectorAdapter):
        OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {"scann": {"cosine": "cosine"}}

    with pytest.raises(UnsupportedCapabilityError) as exc:
        _Stub().opclass("scann", "hamming")

    assert "cosine" in str(exc.value)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_declared_capability_is_a_known_one(name: str) -> None:
    """A capability outside the vocabulary is refused at `supports()`, not ignored.

    Found by running against a real server, which is the wrong place to find it:
    `AlloyDBOmniAdapter` declared `vector_scann` while `CAPABILITIES` did not
    list it, so `capabilities()` looked right, the unit tests asserting its
    contents passed, and `build_index` raised `unknown capability` on the first
    real build. Asserting the dict's contents proves what an adapter *says*;
    only checking it against the vocabulary proves the run can use it.
    """
    declared = set(ADAPTERS[name].factory().capabilities())

    unknown = declared - set(CAPABILITIES)
    assert not unknown, f"{name} declares capabilities not in CAPABILITIES: {sorted(unknown)}"


# ---------------------------------------- a knob the adapter cannot honour
#
# Measured on the droplet, 2026-08-17, and it is the hole a second engine was
# needed to find. Omni's bundled pgvector fork does not register
# `hnsw.ef_search` (zero rows in pg_settings), and AlloyDBOmniAdapter maps
# `num_leaves_to_search`, not `ef_search`. So a sweep of ef_search produced an
# EMPTY mapping, the gate had nothing to verify, and it passed vacuously:
#
#   alloydbomni  ef_search=16   recall@10 = 0.7820
#   alloydbomni  ef_search=256  recall@10 = 0.7820
#
# Three bundle rows labelled 16 / 64 / 256 were one operating point. This is
# `sweep_for`'s own reasoning about exact search -- "sweeping it would produce
# duplicate points under different labels" -- one level further down.


def test_a_requested_knob_the_adapter_cannot_map_is_refused() -> None:
    server = _ServerStub({})
    adapter = _adapter_with(server)

    with pytest.raises(AdapterError, match="cannot apply"):
        adapter.set_search_parameters({"num_leaves_to_search": 500})


def test_the_knobs_an_adapter_declares_are_still_accepted() -> None:
    server = _ServerStub({"hnsw.ef_search": ("64", "session")})
    adapter = _adapter_with(server)

    adapter.set_search_parameters({"ef_search": 64})

    assert adapter.effective_search_parameters() == {"hnsw.ef_search": "64"}


def test_exact_search_passes_no_knobs_and_is_not_refused() -> None:
    """`sweep_for` hands `{}` to a `kind="none"` row; upstream PostgreSQL has no knobs."""
    server = _ServerStub({})
    adapter = PostgresAdapter()
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]

    adapter.set_search_parameters({})

    assert adapter.effective_search_parameters() == {}


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_declared_search_parameter_is_mapped_to_a_guc(name: str) -> None:
    """A declared knob with no mapping would be accepted and do nothing."""
    adapter = ADAPTERS[name].factory()
    declared = getattr(type(adapter), "SEARCH_PARAMETERS", None)
    if declared is None:
        pytest.skip(f"{name} is not a PostgreSQL-family adapter")

    # `probes` maps through ivfflat_lists, which refuses an empty table -- as it
    # should. A real run always has a loaded corpus by the time knobs are set.
    adapter._row_count = 10_000  # type: ignore[attr-defined]

    for knob in declared:
        probe = {knob: 1}
        mapped = adapter._search_guc_mapping(probe)  # type: ignore[attr-defined]
        assert mapped, f"{name} declares {knob!r} but maps it to no GUC"


# --------------------------- TheoDB names its own access methods and knobs
#
# Measured 2026-08-17 against the image the project's own Dockerfile builds
# (theodb:b059, PostgreSQL 18.6, theodb_rs 1.5.0):
#
#   emitted:  CREATE INDEX ... USING hnsw ("embedding" vector_l2_ops) WITH (m = 16)
#   server:   UndefinedObject: access method "hnsw" does not exist
#   pg_am:    theodb_hnsw, theodb_ivfflat
#   pg_opclass: theodb_hnsw_l2_ops, theodb_ivfflat_l2_ops, ...
#
# The `hnsw` alias and `vector_*_ops` do exist -- in the separate `vector` shim
# extension (ADR-0058), which the image creates in `template1`, not in the
# `postgres` database the harness connects to. So the whole indexed half of the
# theodb axis returned INVALID. Tracked as B-064.


def test_theodb_emits_its_own_access_method() -> None:
    adapter = TheoDBAdapter()
    adapter._row_count = 10_000

    name, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="hnsw", parameters={"m": 16}))

    assert "USING theodb_hnsw " in ddl
    assert "USING hnsw " not in ddl
    assert "theodb_hnsw_l2_ops" in ddl
    assert "vector_l2_ops" not in ddl
    # The bundle label stays the index *family*, not the engine's spelling.
    assert name == "bench_vectors_hnsw_l2_idx"


def test_theodb_emits_its_own_ivfflat_access_method() -> None:
    adapter = TheoDBAdapter()
    adapter._row_count = 10_000

    _, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="ivfflat"))

    assert "USING theodb_ivfflat " in ddl
    assert "theodb_ivfflat_l2_ops" in ddl


def test_pgvector_still_emits_the_upstream_access_method() -> None:
    """Regression: only TheoDB renames, and upstream pgvector must not."""
    adapter = PgvectorAdapter()
    adapter._row_count = 10_000

    _, ddl = adapter.index_ddl(SPEC, IndexSpec(kind="hnsw", parameters={"m": 16}))

    assert "USING hnsw " in ddl
    assert "vector_l2_ops" in ddl


def test_theodb_loads_its_library_into_the_session() -> None:
    """Measured: pg_settings holds no `theodb%` row, and no `hnsw.ef_search`
    either, until `LOAD 'theodb_rs'` runs. Sweeping ef_search without it
    searched at the default for every point."""
    server = _ServerStub({})
    adapter = TheoDBAdapter()
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]

    adapter.wait_ready()

    statements = " | ".join(server.executed)
    assert "LOAD 'theodb_rs'" in statements


def test_theodb_maps_ef_search_to_its_own_guc() -> None:
    server = _ServerStub({"theodb_hnsw.ef_search": ("200", "session")})
    adapter = TheoDBAdapter()
    adapter._row_count = 10_000
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]

    adapter.set_search_parameters({"ef_search": 200})

    assert adapter.effective_search_parameters() == {"theodb_hnsw.ef_search": "200"}


def test_theodb_records_the_postgresql_version_too() -> None:
    """The three-way race crosses major versions, and the bundle has to say so.

    Measured 2026-08-17 on one machine: theodb on PostgreSQL 18.6, pgvector on
    17.11, AlloyDB Omni on 17.9. The pgvector and Omni bundles named their
    server; the theodb bundle said only `theodb_rs 1.5.0`, because this override
    replaced the base adapter's version instead of composing with it. The one
    bundle that hid which PostgreSQL it ran on was our own product's.
    """

    class _VersionStub(_ServerStub):
        def fetch_one(
            self, sql: str, parameters: tuple[object, ...] | None = None
        ) -> tuple[object, ...] | None:
            if "version()" in sql:
                return ("PostgreSQL 18.6 (Debian 18.6-1.pgdg12+2) on x86_64",)
            if "pg_extension" in sql:
                return ("1.5.0",)
            return super().fetch_one(sql, parameters)

    adapter = TheoDBAdapter()
    server = _VersionStub({})
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]

    version = adapter.export_config()["version"]

    assert "18.6" in version
    assert "theodb_rs 1.5.0" in version


# ------------------------------- build gets its own time budget, and says so
#
# Measured on 2026-08-17: building an hnsw index over one million SIFT-128
# vectors was cancelled after 61 s by the harness's own
# `statement_timeout = 60_000`, and the run was reported as
# "system under test crashed during the run". The competitor's scann build fitted
# inside the same 60 s, so a single budget silently decided which engines could be
# measured at which scale -- while the report blamed the engine.
#
# A query taking 60 s at k=10 is still a defect worth catching, so the query
# budget stays tight. Building an index is a different risk with a different
# duration, and it gets its own.


def test_the_build_budget_is_larger_than_the_query_budget() -> None:
    config = PostgresConfig()

    assert config.build_timeout_ms > config.statement_timeout_ms


def test_building_an_index_raises_the_budget_and_restores_it() -> None:
    server = _ServerStub({"hnsw.ef_search": ("64", "session")})
    adapter = _adapter_with(server)

    adapter.build_index(SPEC, IndexSpec(kind="hnsw", parameters={"m": 16}))

    statements = [s for s in server.executed if "statement_timeout" in s]
    assert statements, "the build never touched the statement timeout"
    assert f"SET statement_timeout = {PostgresConfig().build_timeout_ms}" in statements[0]
    # Restored afterwards, or every later query would inherit the build's budget
    # and a runaway search would stop being caught.
    assert f"SET statement_timeout = {PostgresConfig().statement_timeout_ms}" in statements[-1]


def test_the_budget_is_restored_even_when_the_build_fails() -> None:
    class _Failing(_ServerStub):
        def execute(self, sql: str, parameters: tuple[object, ...] | None = None) -> None:
            super().execute(sql, parameters)
            if sql.startswith("CREATE INDEX"):
                raise AdapterError("build blew up", context=ErrorContext(phase=Phase.INDEX_BUILD))

    server = _Failing({})
    adapter = _adapter_with(server)

    with pytest.raises(AdapterError):
        adapter.build_index(SPEC, IndexSpec(kind="hnsw", parameters={"m": 16}))

    statements = [s for s in server.executed if "statement_timeout" in s]
    assert f"SET statement_timeout = {PostgresConfig().statement_timeout_ms}" in statements[-1]


# ---------------------- TheoDB's own ScaNN-class path, reachable from the harness
#
# TheoDB has the ScaNN recipe, as reloptions on theodb_ivfflat rather than as an
# access method named scann. Verified on the built image:
#
#   CREATE INDEX ... USING theodb_ivfflat (emb theodb_ivfflat_l2_ops)
#     WITH (lists=20, pq_subspaces=16, pq_bits=4, separate_storage=1, refine=1)
#   -> CREATE INDEX; the planner uses it.
#
# `pq_subspaces` is the anisotropic quantizer (AqQuantizer), `pq_bits=4` is the
# LUT16 pshufb width, `refine=1` plus `separate_storage=1` gives the exact-distance
# second stage, and `aq_threshold` is ScaNN's anisotropic T. The internal name for
# the arc is pg_scann (M75 built the algorithm, M77 the persisted access method).
#
# The rescore pool is `64 * theodb_hnsw.over_fetch` (customscan.rs), so comparing
# our second stage against the competitor's `pre_reordering_num_neighbors` needs
# that knob declared -- otherwise the harness can only sweep probe depth and the
# two rescore pools stay whatever each engine defaults to.


def test_theodb_can_sweep_its_rescore_pool() -> None:
    server = _ServerStub({"theodb_hnsw.over_fetch": ("2", "session")})
    adapter = TheoDBAdapter()
    adapter._row_count = 100_000
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]

    adapter.set_search_parameters({"over_fetch": 2})

    assert adapter.effective_search_parameters() == {"theodb_hnsw.over_fetch": "2"}


def test_theodb_renders_the_pg_scann_reloptions() -> None:
    adapter = TheoDBAdapter()
    adapter._row_count = 100_000

    _, ddl = adapter.index_ddl(
        SPEC,
        IndexSpec(
            kind="ivfflat",
            parameters={
                "lists": 316,
                "pq_subspaces": 16,
                "pq_bits": 4,
                "separate_storage": 1,
                "refine": 1,
            },
        ),
    )

    assert "USING theodb_ivfflat " in ddl
    assert "theodb_ivfflat_l2_ops" in ddl
    for option in ("lists = 316", "pq_subspaces = 16", "pq_bits = 4", "refine = 1"):
        assert option in ddl, option


# ------------------------------------------- bulk load goes through COPY, not INSERT
#
# Measured 2026-08-17 on the droplet: loading one million SIFT-128 vectors took
# **122 s** through `cursor.executemany` in batches of 1000 -- a thousand
# round-trips. The constant driving it is named COPY_BATCH and no `COPY` was ever
# emitted. Load time never enters a published number, but it decides which scales
# are measurable at all, and therefore which claims the project can make.


class _CopyStub(_ServerStub):
    """A server that records what was streamed through COPY."""

    def __init__(self, settings: dict[str, tuple[str, str]]) -> None:
        super().__init__(settings)
        self.copy_statements: list[str] = []
        self.copied_rows: list[tuple[object, ...]] = []
        self.copied_bytes: list[bytes] = []

    def cursor(self) -> _CopyCursor:
        return _CopyCursor(self)


class _CopyCursor:
    def __init__(self, server: _CopyStub) -> None:
        self._server = server

    def __enter__(self) -> _CopyCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def executemany(self, sql: str, batch: list[tuple[object, ...]]) -> None:
        raise AssertionError(f"bulk load fell back to executemany: {sql[:60]}")

    def copy(self, sql: str) -> _CopyWriter:
        self._server.copy_statements.append(sql)
        return _CopyWriter(self._server)


class _CopyWriter:
    def __init__(self, server: _CopyStub) -> None:
        self._server = server

    def __enter__(self) -> _CopyWriter:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def write_row(self, row: tuple[object, ...]) -> None:
        self._server.copied_rows.append(row)

    def write(self, payload: bytes) -> None:
        self._server.copied_bytes.append(payload)


def _copy_adapter(server: _CopyStub) -> PgvectorAdapter:
    adapter = PgvectorAdapter()
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]
    adapter._cursor = server.cursor  # type: ignore[method-assign]
    return adapter


def test_the_vector_load_streams_through_binary_copy() -> None:
    """pgvector's `vector` has a documented binary layout, so the corpus goes over
    the wire with no per-value Python. Measured: the text encoding was 72 of the
    75 seconds a million-vector load took."""
    server = _CopyStub({})
    adapter = _copy_adapter(server)

    adapter.load_dataset(SPEC, np.zeros((5, 8), dtype=np.float32))

    assert server.copy_statements, "no COPY was issued"
    assert "FORMAT BINARY" in server.copy_statements[0]
    stream = b"".join(server.copied_bytes)
    assert stream.startswith(b"PGCOPY\n\xff\r\n\x00")
    assert stream.endswith(struct.pack(">h", -1))
    assert not server.copied_rows, "binary path must not fall back to write_row"


def test_upstream_postgres_still_streams_rows_as_text() -> None:
    """`real[]` has no encoder here, and saying so is better than a wrong one.

    Upstream exact search is the honest floor of a comparison, not a scale target:
    nobody loads a billion vectors into `real[]` to measure them.
    """
    server = _CopyStub({})
    adapter = PostgresAdapter()
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]
    adapter._cursor = server.cursor  # type: ignore[method-assign]

    adapter.load_dataset(SPEC, np.zeros((5, 8), dtype=np.float32))

    assert "FORMAT BINARY" not in server.copy_statements[0]
    assert len(server.copied_rows) == 5


def test_the_analytical_load_streams_through_copy() -> None:
    server = _CopyStub({})
    adapter = _copy_adapter(server)
    table = AnalyticalTable(
        name="bench_analytical_row", columns=("id", "amount", "category", "quantity"), path="row"
    )

    adapter.load_analytical(table, [(0, 1.5, "a", 2), (1, -2.5, "b", 3)])

    assert server.copy_statements[0].startswith("COPY ")
    assert len(server.copied_rows) == 2


def test_every_row_reaches_the_copy_stream() -> None:
    """The count is still proven: speed never replaces the proof that all of it arrived."""
    server = _CopyStub({})
    adapter = _copy_adapter(server)

    outcome = adapter.load_dataset(SPEC, np.zeros((2_500, 8), dtype=np.float32))

    row_bytes = 2 + 8 + (4 + 4 + 8 * 4)
    body = b"".join(server.copied_bytes)[len(BINARY_HEADER) : -len(BINARY_TRAILER)]
    assert len(body) == 2_500 * row_bytes
    assert outcome.rows_expected == 2_500
