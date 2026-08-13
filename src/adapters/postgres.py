"""PostgreSQL-family adapters: upstream PostgreSQL, pgvector, and TheoDB.

The three share a lifecycle and differ in what they can do, so they form a small
hierarchy rather than three copies. Upstream PostgreSQL has no vector type at
all and can only do exact search over ``real[]``; pgvector adds the ``vector``
type with HNSW and IVFFlat; TheoDB adds its own access methods on top.

Four measurement invariants are enforced here rather than trusted:

The index is forced *and* verified. ``SET enable_seqscan = off`` alone proves
nothing -- the planner may still choose a sequential scan, and a benchmark that
believed otherwise would report scan performance under an index's name (I5).

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
from typing import Any, Final

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

# Operator classes per access method and metric. A metric absent from an
# access method's map is unsupported for that method -- never approximated
# with a different one.
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

    def __init__(self, config: PostgresConfig | None = None, **kwargs: Any) -> None:
        self.config = config if config is not None else PostgresConfig(**kwargs)
        self._connection: Any = None
        self._row_count: int = 0
        self._search_parameters: dict[str, Any] = {}
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

    def set_search_parameters(self, parameters: dict[str, Any]) -> None:
        self._search_parameters = dict(parameters)

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
        available = OPCLASSES.get(kind, {})
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

    def index_ddl(self, spec: VectorTableSpec, index: IndexSpec) -> tuple[str, str]:
        """(index name, CREATE INDEX statement) for a configuration."""
        opclass = self.opclass(index.kind, spec.metric)
        name = f"{spec.table}_{index.kind}_{spec.metric}_idx"
        parameters = dict(index.parameters)
        if index.kind == "ivfflat":
            # Derived from the real row count, never from a default.
            parameters.setdefault("lists", ivfflat_lists(self._row_count))
        rendered = ", ".join(f"{key} = {int(value)}" for key, value in sorted(parameters.items()))
        with_clause = f" WITH ({rendered})" if rendered else ""
        ddl = (
            f"CREATE INDEX {_identifier(name)} ON {_identifier(spec.table)} "
            f"USING {index.kind} ({_identifier(spec.embedding_column)} {opclass}){with_clause}"
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

    def set_search_parameters(self, parameters: dict[str, Any]) -> None:
        super().set_search_parameters(parameters)
        for name, value in parameters.items():
            if name == "ef_search":
                self._execute(f"SET hnsw.ef_search = {int(value)}")
            elif name == "probes":
                lists = ivfflat_lists(self._row_count)
                self._execute(f"SET ivfflat.probes = {clamp_probes(int(value), lists)}")

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

    def capabilities(self) -> dict[str, bool]:
        return {
            "vector_exact": True,
            "vector_hnsw": True,
            "vector_ivfflat": True,
            "vector_filtered": True,
            "hybrid": True,
            "lexical": True,
            "columnar": True,
            "parquet": True,
            "graph": True,
            "vectorizer": True,
            "ai_sql": True,
        }

    def export_config(self) -> dict[str, Any]:
        payload = super().export_config()
        row = self._fetch_one(
            "SELECT extversion FROM pg_extension WHERE extname = %s", (self.extension,)
        )
        if row is not None:
            payload["version"] = f"{self.extension} {row[0]}"
        return payload


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
