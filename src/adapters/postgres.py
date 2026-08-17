"""PostgreSQL-family adapters: upstream PostgreSQL, pgvector, and TheoDB.

The three share a lifecycle and differ in what they can do, so they form a small
hierarchy rather than three copies. Upstream PostgreSQL has no vector type at
all and can only do exact search over ``real[]``; pgvector adds the ``vector``
type with HNSW and IVFFlat; TheoDB adds its own access methods on top.

Measurement invariants live here. One of them is **not** enforced, and saying so
is the point:

I5 -- the index forced *and* verified -- is **not in force**. Measured 2026-08-17:
``assert_index_used`` below has no caller anywhere in the package, it raises
``ProgrammingError`` if called (this class overrides ``_query_sql`` to repeat the
distance expression, so the probe binds twice, and the inherited verifier binds
once), and ``SET enable_seqscan = off`` appears in this docstring and nowhere
else in executable code. The harness measures whatever plan the planner chooses.
At the registered suite's size (10 000 x 64) the planner does choose the index on
pgvector, Omni/hnsw and Omni/scann -- verified by EXPLAIN -- so no published
number is retracted; at 200 rows it chose a sequential scan. Tracked as B-063.
This paragraph replaces a claim that this file made for its whole life and that
another item cited as exemplary discipline.

A requested search knob is applied *and* proven in force before a point is
measured -- and a knob the adapter cannot apply is refused rather than ignored.
See :meth:`PostgresAdapter.set_search_parameters`; this one is real, and it is
currently the only apply-then-verify that executes.

Indexes from other configurations are dropped before a point is measured. Two
indexes of the same family on the same column let the planner choose, and one
sweep silently flattens onto the other (I6).

IVFFlat ``lists`` derives from the real row count, and ``probes`` is clamped to
``lists``. A default-derived ``lists`` over a million rows builds a crippled
index; ``probes`` above ``lists`` is a no-op that would report a duplicate point
under a different label (I10, I11).

An access method that has no operator class for the requested metric is
reported as unsupported. Emitting DDL for an opclass that does not exist would
turn a missing capability into a failed run (I13).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

import numpy as np
from theodb_bench.absent import encode, unavailable
from theodb_bench.adapters.base import (
    BuildOutcome,
    IndexSpec,
    KnnQuery,
    KnnResult,
    LoadOutcome,
    SystemAdapter,
    VectorArray,
    VectorTableSpec,
)
from theodb_bench.errors import (
    AdapterError,
    ErrorContext,
    Phase,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)

# S608 is annotated per site below. Table and column names cannot be bound as
# parameters, so they are composed into the SQL -- but only after passing
# _identifier(), which accepts nothing but alphanumerics and underscores.
# Every value is bound.
DEFAULT_DSN: Final[str] = "postgresql:///postgres"
COPY_BATCH: Final[int] = 1000

# Operator classes per access method and metric, in pgvector's convention. A
# metric absent from an access method's map is unsupported for that method --
# never approximated with a different one.
#
# This is pgvector's naming, not a universal one: AlloyDB's scann access method
# names the same three classes `cosine` / `dot_product` / `l2`. An adapter for
# another engine therefore declares its own table (PgvectorAdapter.OPCLASSES)
# rather than inheriting this one.
OPCLASSES: Final[dict[str, dict[str, str]]] = {
    "hnsw": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
    "ivfflat": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
}

DISTANCE_OPERATORS: Final[dict[str, str]] = {"l2": "<->", "ip": "<#>", "cosine": "<=>"}


def _require_driver() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise AdapterError(
            "psycopg is required for the PostgreSQL adapters; "
            "install it with: pip install 'theodb-bench[postgres]'",
            context=ErrorContext(phase=Phase.BOOTSTRAP),
            cause=exc,
        ) from exc
    return psycopg


def ivfflat_lists(row_count: int) -> int:
    """Lists for an IVFFlat index, derived from the real row count.

    pgvector's own guidance is rows/1000 up to a million rows and sqrt(rows)
    beyond. Deriving this from a default row count instead of the real one
    builds a crippled index and calls the result a comparison.
    """
    if row_count <= 0:
        raise AdapterError(
            "cannot size an IVFFlat index for an empty table",
            context=ErrorContext(phase=Phase.INDEX_BUILD),
        )
    lists = row_count // 1000 if row_count <= 1_000_000 else int(math.sqrt(row_count))
    return max(1, lists)


def clamp_probes(probes: int, lists: int) -> int:
    """Clamp probes to lists.

    In pgvector, ``probes > lists`` is a no-op: the extra probes do nothing and
    the point would be reported at a label that does not describe it.
    """
    return max(1, min(probes, lists))


@dataclass
class PostgresConfig:
    """Connection and session settings for a PostgreSQL-family system."""

    dsn: str = DEFAULT_DSN
    statement_timeout_ms: int = 60_000
    session_settings: dict[str, str] = field(default_factory=dict)
    application_name: str = "theodb-bench"


class PostgresAdapter(SystemAdapter):
    """Upstream PostgreSQL: exact search over ``real[]``, no vector type.

    This is the honest floor for a comparison. It is slow, and that is the
    point: it shows what the index is buying.
    """

    system_id = "postgres"

    #: Search parameters this adapter can actually apply. A request naming
    #: anything else is refused rather than silently ignored -- upstream
    #: PostgreSQL has no ANN knobs, and exact search has nothing to tune.
    SEARCH_PARAMETERS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: PostgresConfig | None = None, **kwargs: Any) -> None:
        self.config = config if config is not None else PostgresConfig(**kwargs)
        self._connection: Any = None
        self._row_count: int = 0
        self._search_parameters: dict[str, Any] = {}
        self._effective_search_parameters: dict[str, str] = {}
        self._built_indexes: set[str] = set()

    # ------------------------------------------------------------ capabilities

    def capabilities(self) -> dict[str, bool]:
        return {"vector_exact": True}

    # --------------------------------------------------------------- lifecycle

    def prepare(self) -> None:
        _require_driver()

    def start(self) -> None:
        psycopg = _require_driver()
        try:
            self._connection = psycopg.connect(
                self.config.dsn,
                autocommit=True,
                application_name=self.config.application_name,
            )
        except Exception as exc:  # psycopg raises a family of connection errors
            raise SystemUnavailableError(
                f"could not connect to {self.system_id}",
                context=ErrorContext(phase=Phase.BOOTSTRAP, system=self.system_id),
                cause=exc,
            ) from exc

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._execute("SELECT 1")
                self._apply_session_settings()
                return
            except Exception as exc:
                last = exc
                time.sleep(0.2)
        raise SystemUnavailableError(
            f"{self.system_id} did not become ready within {timeout_seconds}s",
            context=ErrorContext(phase=Phase.BOOTSTRAP, system=self.system_id),
            cause=last,
        )

    def stop(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def cleanup(self) -> None:
        self._built_indexes.clear()

    # ------------------------------------------------------------------- setup

    def _apply_session_settings(self) -> None:
        self._execute(f"SET statement_timeout = {int(self.config.statement_timeout_ms)}")
        for name, value in self.config.session_settings.items():
            self._execute(f"SET {_identifier(name)} = %s", (value,))

    def _cursor(self) -> Any:
        if self._connection is None:
            raise SystemUnavailableError(
                f"{self.system_id} is not connected",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        return self._connection.cursor()

    def _execute(self, sql: str, parameters: tuple[Any, ...] | None = None) -> None:
        with self._cursor() as cursor:
            cursor.execute(sql, parameters)

    def _fetch_all(
        self, sql: str, parameters: tuple[Any, ...] | None = None
    ) -> list[tuple[Any, ...]]:
        with self._cursor() as cursor:
            cursor.execute(sql, parameters)
            rows: list[tuple[Any, ...]] = cursor.fetchall()
        return rows

    def _fetch_one(
        self, sql: str, parameters: tuple[Any, ...] | None = None
    ) -> tuple[Any, ...] | None:
        with self._cursor() as cursor:
            cursor.execute(sql, parameters)
            row: tuple[Any, ...] | None = cursor.fetchone()
        return row

    # -------------------------------------------------------------------- data

    def column_type(self, dimension: int) -> str:
        """Upstream PostgreSQL has no vector type; float4[] is the closest thing."""
        return "real[]"

    def _to_column(self, vector: VectorArray) -> Any:
        return [float(value) for value in vector]

    def load_dataset(self, spec: VectorTableSpec, vectors: VectorArray) -> LoadOutcome:
        table = _identifier(spec.table)
        column = _identifier(spec.embedding_column)
        started = time.perf_counter()
        self._execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        self._execute(
            f"CREATE TABLE {table} (id integer PRIMARY KEY, "
            f"{column} {self.column_type(spec.dimension)} NOT NULL)"
        )
        with self._cursor() as cursor:
            batch: list[tuple[int, Any]] = []
            for index, vector in enumerate(vectors):
                batch.append((index, self._to_column(vector)))
                if len(batch) >= COPY_BATCH:
                    cursor.executemany(f"INSERT INTO {table} (id, {column}) VALUES (%s, %s)", batch)
                    batch.clear()
            if batch:
                cursor.executemany(f"INSERT INTO {table} (id, {column}) VALUES (%s, %s)", batch)

        self._execute(f"ANALYZE {table}")
        row = self._fetch_one(f"SELECT count(*) FROM {table}")
        loaded = int(row[0]) if row else 0
        self._row_count = loaded
        return LoadOutcome(
            seconds=time.perf_counter() - started,
            rows_loaded=loaded,
            rows_expected=int(vectors.shape[0]),
        )

    # ------------------------------------------------------------------ index

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        self.require(index.capability, f"index kind {index.kind!r}")
        if index.kind == "none":
            return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})
        raise UnsupportedCapabilityError(
            f"{self.system_id} cannot build a {index.kind} index",
            context=ErrorContext(phase=Phase.INDEX_BUILD, system=self.system_id),
        )

    def drop_indexes(self, spec: VectorTableSpec) -> None:
        """Remove every index this adapter built on the benchmark table."""
        for name in sorted(self._built_indexes):
            self._execute(f"DROP INDEX IF EXISTS {_identifier(name)}")
        self._built_indexes.clear()

    # ----------------------------------------------------------------- queries

    def _search_guc_mapping(self, parameters: dict[str, Any]) -> dict[str, str]:
        """Which GUC carries each logical parameter, and the literal that will be SENT.

        Subclass hook. The value is what the adapter *sends*, already transformed — the
        clamp on ``probes`` means the sent value legitimately differs from what the caller
        asked for, and the gate must compare against the former. Returning an empty mapping
        means "this system has no session knobs", which upstream PostgreSQL genuinely does
        not.
        """
        return {}

    def set_search_parameters(self, parameters: dict[str, Any]) -> None:
        """Apply the search knobs, then prove the server actually took them.

        The apply-then-verify shape mirrors :meth:`assert_index_used`, and for the same
        reason: setting without verifying proves nothing.

        Why ``pg_settings`` and not ``current_setting`` — measured on PostgreSQL 18, not
        assumed::

            SET nao.existe = 999;                                      -> SET (succeeds)
            SELECT current_setting('nao.existe', true);                 -> 999
            SELECT count(*) FROM pg_settings WHERE name='nao.existe';   -> 0

        An unregistered namespaced GUC is accepted as a *placeholder*: the SET succeeds and
        nothing applies it. ``current_setting`` then hands back the value we wrote, so a gate
        built on it would be a perfect false negative — it would confirm 200 while the engine
        searched at its default. ``pg_settings`` lists only *registered* GUCs, which is
        exactly the distinction that matters.

        A registered GUC is not sufficient either: its ``source`` must have moved off
        ``default``. And absence of the GUC usually means the extension library never loaded
        in this session — measured on TheoDB, ``pg_settings`` holds 0 ``theodb*`` entries
        before ``LOAD 'theodb_rs'`` and 38 after. That is the same condition under which
        AlloyDB's ``scann.num_leaves_to_search`` silently does nothing without
        ``LOAD 'alloydb_scann'``.

        Verifying only the knobs the adapter *mapped* is not enough either, and a second
        engine is what proved it. AlloyDB Omni bundles a fork of pgvector that does not
        register ``hnsw.ef_search`` at all, and its adapter maps
        ``num_leaves_to_search`` instead. A sweep of ``ef_search`` therefore produced an
        empty mapping, the gate had nothing to check, and it passed vacuously --
        publishing three rows labelled 16 / 64 / 256 whose measured recall was identical
        to four decimal places. So a requested knob the adapter does not declare is a
        refusal, for the same reason :meth:`VectorBenchmark.sweep_for` refuses to sweep
        exact search: it would put duplicate points in the table under labels that
        describe operating points nobody ran.
        """
        self._search_parameters = dict(parameters)

        unsupported = sorted(set(parameters) - set(type(self).SEARCH_PARAMETERS))
        if unsupported:
            raise AdapterError(
                f"{self.system_id} cannot apply search parameter(s) "
                f"{', '.join(repr(name) for name in unsupported)}; it understands "
                f"{', '.join(sorted(type(self).SEARCH_PARAMETERS)) or 'none'}. Accepting "
                f"the request and measuring the default would publish this point under a "
                f"label that does not describe it.",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT,
                    system=self.system_id,
                    details={"unsupported": unsupported},
                ),
            )

        mapping = self._search_guc_mapping(parameters)
        for guc, literal in mapping.items():
            self._execute(f"SET {guc} = {literal}")
        self._effective_search_parameters = self._verified_search_settings(mapping)

    def _verified_search_settings(self, mapping: dict[str, str]) -> dict[str, str]:
        """Read each GUC back and refuse anything that did not take effect."""
        effective: dict[str, str] = {}
        for guc, sent in mapping.items():
            try:
                row = self._fetch_one(
                    "SELECT setting, source FROM pg_settings WHERE name = %s", (guc,)
                )
            except AdapterError:
                raise
            except Exception as exc:  # re-raised as a typed error below
                raise AdapterError(
                    f"could not verify {guc}: {exc}. An unreadable setting is not the same "
                    f"as a setting that did not apply, and reporting one as the other would "
                    f"describe a configuration defect where there was an unavailable server.",
                    context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
                ) from exc
            if row is None:
                raise AdapterError(
                    f"{guc} is not a registered setting on this server, so "
                    f"requested={sent} effective=<placeholder>. PostgreSQL accepts a SET on "
                    f"an unregistered namespace without applying it, so the search would run "
                    f"at the engine default while the run reported {sent}. The usual cause is "
                    f"an extension library that never loaded in this session.",
                    context=ErrorContext(
                        phase=Phase.MEASUREMENT,
                        system=self.system_id,
                        details={"setting": guc, "requested": sent},
                    ),
                )
            setting, source = str(row[0]), str(row[1])
            if setting != sent or source == "default":
                raise AdapterError(
                    f"{guc} did not take effect: requested={sent} effective={setting} "
                    f"(source={source}). A registered setting still sitting at its default "
                    f"means the SET was accepted and discarded.",
                    context=ErrorContext(
                        phase=Phase.MEASUREMENT,
                        system=self.system_id,
                        details={"setting": guc, "requested": sent, "effective": setting},
                    ),
                )
            effective[guc] = setting
        return effective

    def effective_search_parameters(self) -> dict[str, str]:
        """The search settings verified in force, keyed by GUC name.

        Distinct from the requested values on purpose: a bundle that publishes the request
        as if it were the measurement is the defect this gate exists to stop.
        """
        return dict(self._effective_search_parameters)

    def distance_expression(self, spec_metric: str, column: str) -> str:
        """SQL computing the distance between the column and a probe.

        Written out in SQL because upstream PostgreSQL has no distance
        operator; the arithmetic is the same one the oracle performs.
        """
        if spec_metric == "l2":
            return f"(SELECT sum((a - b) * (a - b)) FROM unnest({column}, %s::real[]) AS t(a, b))"
        if spec_metric == "ip":
            return f"(-(SELECT sum(a * b) FROM unnest({column}, %s::real[]) AS t(a, b)))"
        if spec_metric == "cosine":
            return (
                f"(1 - (SELECT sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b))) "
                f"FROM unnest({column}, %s::real[]) AS t(a, b)))"
            )
        raise AdapterError(
            f"unknown metric {spec_metric!r}",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def _query_sql(self, query: KnnQuery) -> str:
        table = _identifier(query.table)
        column = _identifier("embedding")
        distance = self.distance_expression(query.metric, column)
        # Deterministic tie-breaking by id: without it, equal distances resolve
        # by physical row order and the top-k boundary shifts between runs.
        return (
            f"SELECT id, {distance} AS distance FROM {table} "
            f"ORDER BY distance, id LIMIT {int(query.k)}"
        )

    def execute(self, query: KnnQuery) -> KnnResult:
        sql = self._query_sql(query)
        probe = self._to_column(query.vector)
        started = time.perf_counter()
        rows = self._fetch_all(sql, (probe,))
        elapsed = time.perf_counter() - started
        return KnnResult(
            ids=tuple(int(row[0]) for row in rows),
            distances=tuple(float(row[1]) for row in rows),
            latency_seconds=elapsed,
        )

    def assert_index_used(self, query: KnnQuery, index_name: str) -> None:
        """Verify from EXPLAIN that the index was actually used.

        Forcing without verifying proves nothing: the planner may ignore the
        hint, and the run would report a sequential scan under an index's name.
        """
        sql = "EXPLAIN (FORMAT JSON) " + self._query_sql(query)
        row = self._fetch_one(sql, (self._to_column(query.vector),))
        if row is None:
            raise AdapterError(
                "EXPLAIN returned no plan",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        plan_text = str(row[0])
        if index_name not in plan_text:
            raise AdapterError(
                f"planner did not use {index_name}; the measurement would describe "
                f"a different access path. Plan: {plan_text[:400]}",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT,
                    system=self.system_id,
                    details={"index": index_name},
                ),
            )

    # ------------------------------------------------------------------ config

    def collect_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {"rows": self._row_count, "indexes": sorted(self._built_indexes)}
        try:
            stats["database_size_bytes"] = self._scalar(
                "SELECT pg_database_size(current_database())"
            )
        except (AdapterError, SystemUnavailableError) as exc:
            # Swallowing this would leave the field simply missing, which reads
            # as "not asked for". Record why it is not there.
            stats["database_size_bytes"] = encode(unavailable(f"pg_database_size failed: {exc}"))
        return stats

    def export_config(self) -> dict[str, Any]:
        """The settings actually in force, read back from the server."""
        wanted = (
            "shared_buffers",
            "work_mem",
            "maintenance_work_mem",
            "max_connections",
            "max_parallel_workers",
            "max_parallel_workers_per_gather",
            "effective_cache_size",
            "synchronous_commit",
            "fsync",
            "full_page_writes",
            "wal_level",
            "checkpoint_timeout",
            "random_page_cost",
            "jit",
        )
        effective: dict[str, Any] = {}
        for name in wanted:
            row = self._fetch_one("SELECT current_setting(%s, true)", (name,))
            if row is not None and row[0] is not None:
                effective[name] = row[0]
        version = self._fetch_one("SELECT version()")
        return {
            "version": str(version[0]) if version else None,
            "effective_configuration": effective,
            "durability": {
                "synchronous_commit": effective.get("synchronous_commit"),
                "fsync": effective.get("fsync"),
            },
        }

    def _scalar(self, sql: str) -> Any:
        row = self._fetch_one(sql)
        return row[0] if row else None


class PgvectorAdapter(PostgresAdapter):
    """PostgreSQL with pgvector: the ``vector`` type, HNSW and IVFFlat."""

    system_id = "pgvector"
    extension = "vector"

    #: Operator classes this engine names, per access method and metric.
    #: A subclass whose engine uses a different convention overrides it; the
    #: lookup in `opclass` reads it off the concrete class, so inheriting the
    #: wrong table is not possible by accident.
    OPCLASSES: ClassVar[dict[str, dict[str, str]]] = OPCLASSES

    #: `ef_search` for HNSW, `probes` for IVFFlat -- both registered GUCs of the
    #: extension, verified in force before a point is measured.
    SEARCH_PARAMETERS: ClassVar[frozenset[str]] = frozenset({"ef_search", "probes"})

    #: Engine access-method name per index family. The family is the *label* a
    #: bundle reports (`hnsw`); the access method is what the engine calls its
    #: implementation, and the two are not always the same word. Empty means the
    #: label is already the engine's name, which is true for upstream pgvector.
    ACCESS_METHODS: ClassVar[dict[str, str]] = {}

    #: Library to LOAD into the session, when the extension's GUCs are only
    #: registered once it is loaded. Measured on both TheoDB and AlloyDB Omni:
    #: `pg_settings` lists none of the extension's settings before the LOAD, so
    #: every `SET` is a placeholder the server accepts and ignores. None means
    #: the extension registers its GUCs without one.
    library: ClassVar[str | None] = None

    def capabilities(self) -> dict[str, bool]:
        return {
            "vector_exact": True,
            "vector_hnsw": True,
            "vector_ivfflat": True,
            "vector_filtered": True,
        }

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        super().wait_ready(timeout_seconds)
        self._execute(f"CREATE EXTENSION IF NOT EXISTS {_identifier(self.extension)} CASCADE")
        if type(self).library is not None:
            # CREATE EXTENSION registers the access method in the catalog; the
            # LOAD registers the extension's GUCs in *this backend*. They are
            # different acts, and only the second one makes `SET` mean anything.
            self._execute(f"LOAD {_literal(str(type(self).library))}")

    def export_config(self) -> dict[str, Any]:
        """Server configuration, with the server *and* extension versions named.

        Both halves are read from the server, never inferred from an image tag.
        A three-way race measured on one machine on 2026-08-17 ran TheoDB on
        PostgreSQL 18.6, pgvector on 17.11 and AlloyDB Omni on 17.9 -- the
        comparison crosses a major version, and a bundle that records only the
        extension hides that from every reader of the result. It is the same
        reason the tag is not trusted: the published Omni image says `latest`
        and serves 17.

        A server that will not answer gets nothing invented for it.
        """
        payload = super().export_config()
        parts: list[str] = []

        server = payload.get("version")
        if server:
            # Trim at " on <arch>": the platform triple is already in the
            # environment record, and repeating it buries the version.
            parts.append(str(server).split(" on ")[0])

        extension = self._fetch_one(
            "SELECT extversion FROM pg_extension WHERE extname = %s", (self.extension,)
        )
        if extension is not None and extension[0]:
            parts.append(f"{self.extension} {extension[0]}")

        if parts:
            payload["version"] = " / ".join(parts)
        else:
            payload.pop("version", None)
        return payload

    def column_type(self, dimension: int) -> str:
        return f"vector({int(dimension)})"

    def _to_column(self, vector: VectorArray) -> Any:
        # pgvector's text input format, which is what the driver sends without
        # the extension's own type adapters registered.
        return "[" + ",".join(repr(float(value)) for value in np.asarray(vector)) + "]"

    def distance_expression(self, spec_metric: str, column: str) -> str:
        operator = DISTANCE_OPERATORS.get(spec_metric)
        if operator is None:
            raise AdapterError(
                f"unknown metric {spec_metric!r}",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        return f"{column} {operator} %s::vector"

    def opclass(self, kind: str, metric: str) -> str:
        """Operator class for an access method and metric, or unsupported."""
        available = type(self).OPCLASSES.get(kind, {})
        opclass = available.get(metric)
        if opclass is None:
            raise UnsupportedCapabilityError(
                f"{self.system_id} has no {kind} operator class for metric {metric!r}; "
                f"available: {', '.join(sorted(available)) or 'none'}",
                context=ErrorContext(
                    phase=Phase.INDEX_BUILD,
                    system=self.system_id,
                    details={"index": kind, "metric": metric},
                ),
            )
        return opclass

    def _render_reloption(self, key: str, value: Any) -> str:
        """Render one `WITH (...)` value, by type.

        Not every index parameter is an integer. AlloyDB's scann access method
        takes `quantizer='sq8'`, and rendering it through `int()` raises a bare
        ValueError with no phase, no system and no option name -- a failure the
        caller cannot act on.

        A type this does not know is refused rather than coerced: a benchmark
        definition that reached here with a list or a dict is a broken
        definition, and silently stringifying it would put an unintended index
        configuration into a published measurement.
        """
        if isinstance(value, bool):
            # Before the int branch: bool is a subclass of int in Python, and
            # `WITH (opt = 1)` is not what PostgreSQL wants for a boolean.
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return _literal(value)
        raise AdapterError(
            f"index parameter {key!r} has value {value!r} of type "
            f"{type(value).__name__}, which cannot be rendered as a reloption; "
            "expected an int, a bool or a str",
            context=ErrorContext(
                phase=Phase.INDEX_BUILD,
                system=self.system_id,
                details={"parameter": key},
            ),
        )

    def index_ddl(self, spec: VectorTableSpec, index: IndexSpec) -> tuple[str, str]:
        """(index name, CREATE INDEX statement) for a configuration."""
        opclass = self.opclass(index.kind, spec.metric)
        name = f"{spec.table}_{index.kind}_{spec.metric}_idx"
        parameters = dict(index.parameters)
        if index.kind == "ivfflat":
            # Derived from the real row count, never from a default.
            parameters.setdefault("lists", ivfflat_lists(self._row_count))
        rendered = ", ".join(
            f"{key} = {self._render_reloption(key, value)}"
            for key, value in sorted(parameters.items())
        )
        with_clause = f" WITH ({rendered})" if rendered else ""
        access_method = type(self).ACCESS_METHODS.get(index.kind, index.kind)
        ddl = (
            f"CREATE INDEX {_identifier(name)} ON {_identifier(spec.table)} "
            f"USING {access_method} ({_identifier(spec.embedding_column)} {opclass}){with_clause}"
        )
        return name, ddl

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        self.require(index.capability, f"index kind {index.kind!r}")
        if index.kind == "none":
            return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})

        name, ddl = self.index_ddl(spec, index)
        started = time.perf_counter()
        self._execute(ddl)
        elapsed = time.perf_counter() - started
        self._built_indexes.add(name)
        size = self._scalar(f"SELECT pg_relation_size({_literal(name)})")
        return BuildOutcome(
            seconds=elapsed,
            index_size_bytes=int(size) if size is not None else None,
            parameters_in_force=dict(index.parameters),
        )

    def _search_guc_mapping(self, parameters: dict[str, Any]) -> dict[str, str]:
        """The two knobs this adapter owns, and the literal each one sends.

        ``probes`` is clamped to the list count derived from the real row count, so the sent
        literal legitimately differs from the request. The gate compares against what is sent
        — comparing against the request would make the harness's own sizing rule trip it.
        """
        mapping: dict[str, str] = {}
        for name, value in parameters.items():
            if name == "ef_search":
                mapping["hnsw.ef_search"] = str(int(value))
            elif name == "probes":
                lists = ivfflat_lists(self._row_count)
                mapping["ivfflat.probes"] = str(clamp_probes(int(value), lists))
        return mapping

    def _query_sql(self, query: KnnQuery) -> str:
        table = _identifier(query.table)
        column = _identifier("embedding")
        distance = self.distance_expression(query.metric, column)
        return (
            f"SELECT id, {distance} AS distance FROM {table} "
            f"ORDER BY {distance}, id LIMIT {int(query.k)}"
        )

    def execute(self, query: KnnQuery) -> KnnResult:
        sql = self._query_sql(query)
        probe = self._to_column(query.vector)
        started = time.perf_counter()
        # The ORDER BY repeats the distance expression, so the probe is bound twice.
        rows = self._fetch_all(sql, (probe, probe))
        elapsed = time.perf_counter() - started
        return KnnResult(
            ids=tuple(int(row[0]) for row in rows),
            distances=tuple(float(row[1]) for row in rows),
            latency_seconds=elapsed,
        )


class TheoDBAdapter(PgvectorAdapter):
    """TheoDB: PostgreSQL 18 with the theodb_rs extension."""

    system_id = "theodb"
    extension = "theodb_rs"

    #: Measured on the image this project's own Dockerfile builds (PostgreSQL
    #: 18.6, theodb_rs 1.5.0): `pg_am` holds `theodb_hnsw` and `theodb_ivfflat`,
    #: and there is no bare `hnsw`. The bare name and the `vector_*_ops` classes
    #: come from the separate `vector` compatibility shim (ADR-0058), which the
    #: image creates in `template1` and not in the `postgres` database a client
    #: connects to by default -- so inheriting pgvector's spelling made every
    #: indexed row of this axis fail with `access method "hnsw" does not exist`.
    #:
    #: The shim's access method uses the *same* handler, so measuring through it
    #: would measure the same engine. Naming the native surface is what makes the
    #: bundle say which surface was exercised.
    ACCESS_METHODS: ClassVar[dict[str, str]] = {
        "hnsw": "theodb_hnsw",
        "ivfflat": "theodb_ivfflat",
    }

    OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {
        "hnsw": {
            "l2": "theodb_hnsw_l2_ops",
            "ip": "theodb_hnsw_ip_ops",
            "cosine": "theodb_hnsw_cosine_ops",
        },
        "ivfflat": {
            "l2": "theodb_ivfflat_l2_ops",
            "ip": "theodb_ivfflat_ip_ops",
            "cosine": "theodb_ivfflat_cosine_ops",
        },
    }

    #: Measured: a fresh session holds zero `theodb%` rows in `pg_settings`, and
    #: no `hnsw.ef_search` either, until this loads. Every swept `ef_search`
    #: before this line was a placeholder, and the search ran at the default.
    library: ClassVar[str | None] = "theodb_rs"

    def capabilities(self) -> dict[str, bool]:
        """What this adapter can actually exercise, not what TheoDB can do.

        TheoDB the database has hybrid search, a columnar table access method,
        Parquet I/O, persisted-CSR graph traversal and a background vectorizer.
        This adapter reaches none of them yet: the lifecycle methods those
        surfaces need (load_documents, traverse, load_analytical,
        insert_document and the rest) are not implemented here.

        Declaring a capability the adapter cannot exercise would put a false
        claim into every system.json and into `theodb-bench list`. A capability
        is a statement about this code path, not about the product.
        """
        return {
            "vector_exact": True,
            "vector_hnsw": True,
            "vector_ivfflat": True,
            "vector_filtered": True,
        }

    def _search_guc_mapping(self, parameters: dict[str, Any]) -> dict[str, str]:
        """TheoDB's own GUC namespaces, matching the access methods it registers.

        The engine also registers `hnsw.ef_search` and `ivfflat.probes` as
        compatibility aliases, and both work. The native names are used because
        this adapter builds the native access methods, and a bundle should not
        report a compatibility spelling for a run that exercised the engine's own
        surface.
        """
        mapping: dict[str, str] = {}
        for name, value in parameters.items():
            if name == "ef_search":
                mapping["theodb_hnsw.ef_search"] = str(int(value))
            elif name == "probes":
                lists = ivfflat_lists(self._row_count)
                mapping["theodb_ivfflat.probes"] = str(clamp_probes(int(value), lists))
        return mapping


# --------------------------------------------------------------------- helpers


def _identifier(name: str) -> str:
    """Quote an SQL identifier, rejecting anything that is not one.

    Identifiers here come from benchmark definitions rather than from user
    input, but a benchmark definition is still data, and data does not get to
    write SQL.
    """
    if not name or not all(char.isalnum() or char == "_" for char in name):
        raise AdapterError(
            f"invalid SQL identifier {name!r}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    return f'"{name}"'


def _literal(value: str) -> str:
    """Quote a string literal for the few places a parameter cannot be bound."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"
