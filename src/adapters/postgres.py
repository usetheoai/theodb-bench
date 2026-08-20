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

import contextlib
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Final

import numpy as np
from theodb_bench.absent import encode, unavailable
from theodb_bench.adapters.base import (
    AnalyticalQuery,
    AnalyticalResult,
    AnalyticalTable,
    BatchQuery,
    BatchResult,
    BuildOutcome,
    Document,
    DocumentTableSpec,
    GraphSpec,
    HybridQuery,
    IndexSpec,
    KnnQuery,
    KnnResult,
    LexicalQuery,
    LoadOutcome,
    RankedResult,
    SystemAdapter,
    TraversalQuery,
    TraversalResult,
    VectorArray,
    VectorTableSpec,
)
from theodb_bench.copy_binary import BINARY_HEADER, BINARY_TRAILER, encode_vector_rows
from theodb_bench.errors import (
    AdapterError,
    ErrorContext,
    Phase,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)
from theodb_bench.streaming import CorpusSource, chunk_source

# S608 is annotated per site below. Table and column names cannot be bound as
# parameters, so they are composed into the SQL -- but only after passing
# _identifier(), which accepts nothing but alphanumerics and underscores.
# Every value is bound.
DEFAULT_DSN: Final[str] = "postgresql:///postgres"

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

    #: Budget for a *query*. Deliberately tight: a k=10 search that takes a minute
    #: is a defect worth catching, not a measurement worth keeping.
    statement_timeout_ms: int = 60_000

    #: Where `write_parquet` puts files and `read_parquet` reads them. Written by
    #: the *server* process, so it must be writable by the database user rather
    #: than by whoever runs the harness.
    parquet_directory: str = "/var/lib/postgresql/theodb-bench-parquet"

    #: Budget for building an index, which is a different risk with a different
    #: duration. Measured: an hnsw build over one million SIFT-128 vectors was
    #: cancelled at 61 s under the query budget, and the run was then reported as
    #: the system under test crashing. The competitor's scann build fitted inside
    #: 60 s, so one shared budget silently decided which engines were measurable at
    #: which scale while the report blamed the engine. An hour is generous enough
    #: for a billion-scale build and still catches a genuinely hung one.
    build_timeout_ms: int = 3_600_000
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
        #: Tables whose BM25 index was built in this session. Searching one that
        #: was never built is refused rather than answered with zero rows.
        self._lexical_built: set[str] = set()
        #: Edge relations whose persisted CSR was folded in this session.
        self._graphs_built: set[str] = set()

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

    #: Rows per binary COPY chunk. Chosen so one chunk of a 1536-dimension corpus
    #: stays near 12 MB: large enough that the per-chunk overhead disappears,
    #: small enough that a corpus larger than memory streams rather than
    #: materialising.
    COPY_CHUNK_ROWS: ClassVar[int] = 50_000

    def _supports_binary_copy(self) -> bool:
        """Whether this adapter's vector column has a binary encoder.

        Upstream PostgreSQL stores `real[]`, whose binary array representation is
        a different and more elaborate layout than pgvector's. It is not encoded
        here because upstream exact search is the honest floor of a comparison
        rather than a scale target -- nobody loads a billion vectors into
        `real[]` to measure them.
        """
        return False

    def _copy_vectors(self, table: str, column: str, vectors: VectorArray) -> None:
        """Stream the corpus into the table, binary when the adapter can.

        Measured on a million SIFT-128 vectors: batched INSERTs 122 s, text COPY
        75 s of which 72 s was the Python text encoding, binary COPY streams the
        same data with no per-value Python at all.
        """
        if not self._supports_binary_copy():
            with (
                self._cursor() as cursor,
                cursor.copy(f"COPY {table} (id, {column}) FROM STDIN") as copy,
            ):
                for index, vector in enumerate(vectors):
                    copy.write_row((index, self._to_column(vector)))
            return

        chunk = type(self).COPY_CHUNK_ROWS
        with (
            self._cursor() as cursor,
            cursor.copy(f"COPY {table} (id, {column}) FROM STDIN WITH (FORMAT BINARY)") as copy,
        ):
            copy.write(BINARY_HEADER)
            for start in range(0, int(vectors.shape[0]), chunk):
                copy.write(encode_vector_rows(vectors[start : start + chunk], start_id=start))
            copy.write(BINARY_TRAILER)

    def load_dataset_streaming(
        self, spec: VectorTableSpec, source: CorpusSource, *, chunk_rows: int = 50_000
    ) -> LoadOutcome:
        """Load a corpus that never has to be resident.

        The array form takes 512 GB of RAM for a billion 128-dimension vectors.
        This one holds one chunk at a time, so the ceiling becomes the disk rather
        than the memory. The row ids come from the chunk rather than a counter, so a
        resumed load does not renumber the rows the dataset's neighbour lists point
        at.

        Binary COPY only: the text path would put the per-value Python cost back,
        and at this scale that cost is the whole load.
        """
        if not self._supports_binary_copy():
            raise UnsupportedCapabilityError(
                f"{self.system_id} has no binary encoder for its vector column, and "
                f"streaming a corpus through the text path would reinstate the "
                f"per-value encoding that makes this scale unreachable",
                context=ErrorContext(phase=Phase.DATASET_LOAD, system=self.system_id),
            )

        table = _identifier(spec.table)
        column = _identifier(spec.embedding_column)
        started = time.perf_counter()
        self._execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        self._execute(
            f"CREATE TABLE {table} (id integer PRIMARY KEY, "
            f"{column} {self.column_type(spec.dimension)} NOT NULL"
            f"{self._filter_column(spec)})"
        )
        with (
            self._under_bulk_budget(),
            self._cursor() as cursor,
            cursor.copy(f"COPY {table} (id, {column}) FROM STDIN WITH (FORMAT BINARY)") as copy,
        ):
            copy.write(BINARY_HEADER)
            for start, block in chunk_source(source, chunk_rows):
                copy.write(encode_vector_rows(block, start_id=start))
            copy.write(BINARY_TRAILER)

        self._execute(f"ANALYZE {table}")
        counted = self._fetch_one(f"SELECT count(*) FROM {table}")
        loaded = int(counted[0]) if counted else 0
        self._row_count = loaded
        return LoadOutcome(
            seconds=time.perf_counter() - started,
            rows_loaded=loaded,
            rows_expected=source.row_count,
        )

    def postmaster_start_time(self) -> datetime | None:
        """When the server last started, from the server's own clock.

        Used only after a run has already failed as unreachable, to say whether
        the system went down and came back or whether the path to it broke. The
        two look identical from the client and are different findings.

        Both timestamps come from the server so the answer does not depend on
        drift between two machines, and a probe that cannot connect returns
        `None` rather than a guess -- the system is known to be in trouble, and
        an unanswered diagnostic is an honest outcome.
        """
        try:
            row = self._fetch_one("SELECT pg_postmaster_start_time()")
        except Exception:  # the system is already known to be unreachable
            return None
        if not row or row[0] is None:
            return None
        value = row[0]
        return value if isinstance(value, datetime) else None

    @contextlib.contextmanager
    def _under_bulk_budget(self) -> Iterator[None]:
        """Run bulk work under the wide budget, and put the query budget back.

        Bulk work here is an index build or a dataset load. They look different
        and are the same kind of thing: unmeasured work whose duration is a
        property of the scale rather than a symptom, and which the tight query
        budget was never meant to bound. That budget exists so a runaway *search*
        cannot stall a run, and applying it to data movement is how a 20 000 000
        vector load aborted at `COPY bench_vectors, line 4569000` — correctly
        classified as `budget_exceeded` rather than a crash, because the system
        under test had not failed.

        The restore runs even when the body raises. Leaving the wide budget in
        place would mean the next measured query is effectively unbounded, which
        is the safety property being borrowed from, not discarded.
        """
        self._execute(f"SET statement_timeout = {int(self.config.build_timeout_ms)}")
        try:
            yield
        finally:
            self._restore_query_budget()

    def _restore_query_budget(self) -> None:
        """Put the query budget back, without masking why the build failed.

        A dead connection makes this fail too, and the original exception is the
        one worth propagating -- so the restore is allowed to fail silently here
        and only here. Every later statement on a dead connection raises anyway.
        """
        with contextlib.suppress(Exception):  # see the docstring
            self._execute(f"SET statement_timeout = {int(self.config.statement_timeout_ms)}")

    def load_dataset(self, spec: VectorTableSpec, vectors: VectorArray) -> LoadOutcome:
        table = _identifier(spec.table)
        column = _identifier(spec.embedding_column)
        started = time.perf_counter()
        self._execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        self._execute(
            f"CREATE TABLE {table} (id integer PRIMARY KEY, "
            f"{column} {self.column_type(spec.dimension)} NOT NULL"
            f"{self._filter_column(spec)})"
        )
        # Streamed through COPY rather than batched INSERTs. Measured before the
        # change: one million SIFT-128 vectors took 122 s as a thousand
        # `executemany` round-trips of a thousand rows. Load time never enters a
        # published number, but it decides which scales are measurable at all --
        # and a benchmark that cannot load a billion vectors cannot measure one.
        #
        # Text format, not binary, and the reason is the vector type: a binary COPY
        # needs a registered binary dumper per type, and psycopg has none for
        # `vector` without an extra dependency that would help only the
        # pgvector-family adapters. Text COPY is one statement for the whole load
        # on every adapter, and `_to_column` already produces the text each engine
        # expects.
        with self._under_bulk_budget():
            self._copy_vectors(table, column, vectors)

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

    def _build_session_settings(self, index: IndexSpec) -> dict[str, str]:
        """Session settings an index needs in force *while it is being built*.

        Distinct from the search knobs: those are applied per operating point,
        after the index exists. A build-time switch applied afterwards changes
        nothing about the index that was already written, so an adapter that
        treated one as the other would build one structure and label it another.
        """
        return {}

    def _apply_build_session(self, index: IndexSpec) -> None:
        mapping = self._build_session_settings(index)
        if not mapping:
            return
        for guc, literal in mapping.items():
            self._execute(f"SET {guc} = {literal}")
        # Verified in force, exactly as the search knobs are.
        self._verified_search_settings(mapping)

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        self.require(index.capability, f"index kind {index.kind!r}")
        if index.kind == "none":
            return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})
        self._apply_build_session(index)
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

    def _filter_column(self, spec: VectorTableSpec) -> str:
        """The generated tenant column, when the workload filters.

        Generated from the id rather than loaded as data: the partition is a
        property of the row number, both sides agree on it by construction, and
        the load path does not have to carry a second column through binary
        COPY to say something the id already says.
        """
        if spec.filter_cardinality is None:
            return ""
        return (
            f", tenant text GENERATED ALWAYS AS "
            f"({_literal('t')} || (id % {int(spec.filter_cardinality)})::text) STORED"
        )

    def knn_sql(self, query: KnnQuery) -> str:
        """The SQL a k-NN probe becomes. Public so a test can read it."""
        return self._query_sql(query)

    #: Whether `ORDER BY` repeats the distance expression instead of naming its
    #: alias. pgvector's index is only chosen for the repeated form, and the
    #: probe is then bound twice. Declared rather than reimplemented: the two
    #: spellings existed as two `_query_sql` bodies, and a filter added to one
    #: silently missed the other.
    ORDER_REPEATS_DISTANCE: ClassVar[bool] = False

    def _query_sql(self, query: KnnQuery) -> str:
        table = _identifier(query.table)
        column = _identifier("embedding")
        distance = self.distance_expression(query.metric, column)
        order = distance if self.ORDER_REPEATS_DISTANCE else "distance"
        # A filtered probe restricts before ordering. The clause is what makes a
        # graph index show whether its edges respect the filter: a system that
        # descends across tenants answers fast and wrong, and only recall against
        # a filtered oracle tells that apart from answering fast and right.
        where = f"WHERE tenant = {_literal(query.tenant)} " if query.tenant is not None else ""
        # Deterministic tie-breaking by id: without it, equal distances resolve
        # by physical row order and the top-k boundary shifts between runs.
        return (
            f"SELECT id, {distance} AS distance FROM {table} "
            f"{where}ORDER BY {order}, id LIMIT {int(query.k)}"
        )

    def execute(self, query: KnnQuery) -> KnnResult:
        sql = self._query_sql(query)
        probe = self._to_column(query.vector)
        # Bound once or twice, from the same declaration that shaped the ORDER BY:
        # deriving it keeps the two from drifting apart.
        parameters = (probe, probe) if self.ORDER_REPEATS_DISTANCE else (probe,)
        started = time.perf_counter()
        rows = self._fetch_all(sql, parameters)
        elapsed = time.perf_counter() - started
        return KnnResult(
            ids=tuple(int(row[0]) for row in rows),
            distances=tuple(float(row[1]) for row in rows),
            latency_seconds=elapsed,
        )

    def execute_batch(self, query: BatchQuery) -> BatchResult:
        """Several probes in one round trip, as a single UNION ALL statement.

        One statement, one round trip: that is the whole measurement. Looping
        over `execute` here would report round-trip savings that never happened,
        which is why the base class refuses rather than looping.

        Each probe keeps its own `LIMIT k` inside a lateral subquery, because a
        single ORDER BY over the union would return the k best *across* probes
        rather than k for each.
        """
        if not query.vectors:
            return BatchResult(ids=(), distances=(), latency_seconds=0.0)

        table = _identifier(query.table)
        column = _identifier("embedding")
        distance = self.distance_expression(query.metric, column)
        order = distance if self.ORDER_REPEATS_DISTANCE else "distance"

        branches: list[str] = []
        parameters: list[Any] = []
        tenants = query.tenants or tuple(None for _ in query.vectors)
        for probe_index, (vector, tenant) in enumerate(zip(query.vectors, tenants, strict=False)):
            where = f"WHERE tenant = {_literal(tenant)} " if tenant is not None else ""
            branches.append(
                f"(SELECT {probe_index} AS probe, id, {distance} AS distance FROM {table} "
                f"{where}ORDER BY {order}, id LIMIT {int(query.k)})"
            )
            column_value = self._to_column(vector)
            parameters.extend(
                (column_value, column_value) if self.ORDER_REPEATS_DISTANCE else (column_value,)
            )

        sql = " UNION ALL ".join(branches) + " ORDER BY probe, distance, id"
        started = time.perf_counter()
        rows = self._fetch_all(sql, tuple(parameters))
        elapsed = time.perf_counter() - started

        grouped: list[list[tuple[int, float]]] = [[] for _ in query.vectors]
        for row in rows:
            grouped[int(row[0])].append((int(row[1]), float(row[2])))
        return BatchResult(
            ids=tuple(tuple(i for i, _ in probe) for probe in grouped),
            distances=tuple(tuple(d for _, d in probe) for probe in grouped),
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

    # ----------------------------------------------------------- analytical

    #: Table access method per analytical execution path. `row` is heap, which
    #: PostgreSQL uses without being told. A path absent from this map is not
    #: supported by the adapter and is refused rather than silently stored as heap
    #: -- measuring heap under the columnar label is the whole defect this
    #: surface exists to avoid.
    ANALYTICAL_PATHS: ClassVar[dict[str, str | None]] = {"row": None}

    ANALYTICAL_SQL: ClassVar[dict[str, str]] = {
        "total_rows": "SELECT count(*) FROM {table}",
        "sum_amount": "SELECT sum(amount) FROM {table}",
        "group_by_category": (
            "SELECT category, sum(amount) FROM {table} GROUP BY category ORDER BY category"
        ),
        # The predicate is the oracle's, read from `expected_answer` rather than
        # invented: it filters `category = 'a' AND amount > 0`. An earlier version
        # of this line used `quantity < 24`, copied from a published TPC-H-shaped
        # query, and the benchmark's own correctness check caught it -- the run
        # came back INVALID rather than reporting a fast wrong number.
        "filtered_sum": ("SELECT sum(amount) FROM {table} WHERE category = 'a' AND amount > 0"),
    }

    def _analytical_column_types(self) -> str:
        return "id integer, amount double precision, category text, quantity integer"

    def _require_analytical_path(self, path: str) -> str | None:
        if path not in type(self).ANALYTICAL_PATHS:
            raise UnsupportedCapabilityError(
                f"{self.system_id} has no {path!r} analytical execution path; "
                f"it has {', '.join(sorted(type(self).ANALYTICAL_PATHS))}",
                context=ErrorContext(
                    phase=Phase.DATASET_LOAD,
                    system=self.system_id,
                    details={"path": path},
                ),
            )
        return type(self).ANALYTICAL_PATHS[path]

    def load_analytical(
        self, table: AnalyticalTable, rows: Sequence[tuple[Any, ...]]
    ) -> LoadOutcome:
        """Load the same rows into one execution path.

        The access method is declared per adapter, never derived from the path
        name: `columnar` is `theodb_columnar` on TheoDB and a heap table plus a
        cache registration on AlloyDB Omni. Two mechanisms, one label.
        """
        access_method = self._require_analytical_path(table.path)
        name = _identifier(table.name)
        using = f" USING {access_method}" if access_method else ""

        started = time.perf_counter()
        self._execute(f"DROP TABLE IF EXISTS {name} CASCADE")
        self._execute(f"CREATE TABLE {name} ({self._analytical_column_types()}){using}")
        columns = ", ".join(_identifier(column) for column in table.columns)
        with (
            self._cursor() as cursor,
            cursor.copy(f"COPY {name} ({columns}) FROM STDIN") as copy,
        ):
            for row in rows:
                copy.write_row(row)
        self._execute(f"ANALYZE {name}")
        self._apply_analytical_session(table)
        self._after_analytical_load(table)

        counted = self._fetch_one(f"SELECT count(*) FROM {name}")
        loaded = int(counted[0]) if counted else 0
        return LoadOutcome(
            seconds=time.perf_counter() - started,
            rows_loaded=loaded,
            rows_expected=len(rows),
        )

    def _after_analytical_load(self, table: AnalyticalTable) -> None:
        """Hook for a path whose residency is a separate act from storing the rows.

        Heap and a columnar table access method need nothing here: writing the
        rows *is* putting them in the path. A cache-based engine does not, and
        overrides this.
        """

    #: Session settings a path needs to actually be that path, per path name.
    #: Applied and then read back from `pg_settings`, exactly as the search knobs
    #: are: a GUC that ships off and silently stays off is how a measurement ends
    #: up describing a configuration nobody ran.
    ANALYTICAL_SESSION_SETTINGS: ClassVar[dict[str, dict[str, str]]] = {}

    #: Plan fragment that proves the path was used, per path name. A path absent
    #: from this map is proven by the catalog alone, which is correct for heap:
    #: there is no other path a heap table could take.
    ANALYTICAL_PLAN_MARKERS: ClassVar[dict[str, str]] = {}

    def _apply_analytical_session(self, table: AnalyticalTable) -> None:
        mapping = type(self).ANALYTICAL_SESSION_SETTINGS.get(table.path, {})
        if not mapping:
            return
        for guc, literal in mapping.items():
            self._execute(f"SET {guc} = {literal}")
        # Same verification the search knobs get, and for the same reason.
        self._verified_search_settings(mapping)

    def _analytical_query_sql(self, table: AnalyticalTable, query: AnalyticalQuery) -> str:
        template = type(self).ANALYTICAL_SQL.get(query.id)
        if template is None:
            raise AdapterError(
                f"unknown analytical query {query.id!r}; "
                f"{self.system_id} knows {', '.join(sorted(type(self).ANALYTICAL_SQL))}",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT,
                    system=self.system_id,
                    details={"query": query.id},
                ),
            )
        return template.format(table=_identifier(table.name))

    def execute_analytical(
        self, table: AnalyticalTable, query: AnalyticalQuery
    ) -> AnalyticalResult:
        sql = self._analytical_query_sql(table, query)
        started = time.perf_counter()
        rows = self._fetch_all(sql)
        elapsed = time.perf_counter() - started
        return AnalyticalResult(
            rows=tuple(tuple(row) for row in rows),
            wall_seconds=elapsed,
        )

    def assert_analytical_path(
        self, table: AnalyticalTable, query: AnalyticalQuery | None = None
    ) -> None:
        """Prove the rows are really in the path the label claims.

        The default proof is the catalog: `pg_class.relam` names the table access
        method, and it cannot disagree with where the rows are. An engine whose
        columnar surface is a cache needs a different proof and overrides this.
        """
        expected = self._require_analytical_path(table.path)
        if expected is None:
            return
        row = self._fetch_one(
            "SELECT a.amname FROM pg_class c JOIN pg_am a ON a.oid = c.relam WHERE c.relname = %s",
            (table.name,),
        )
        actual = str(row[0]) if row and row[0] else "<absent>"
        if actual != expected:
            raise AdapterError(
                f"{table.name} claims path {table.path!r} but its access method is "
                f"{actual!r}, not {expected!r}. Measuring it would report "
                f"{actual} timings under the {table.path} label.",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT,
                    system=self.system_id,
                    details={"path": table.path, "access_method": actual},
                ),
            )
        self._assert_plan_uses_the_path(table, query)

    def _assert_plan_uses_the_path(
        self, table: AnalyticalTable, query: AnalyticalQuery | None = None
    ) -> None:
        """Residency proves where the rows are; only the plan proves what ran.

        Measured on the built image at one million rows, same table and query:
        with `theodb.enable_columnar_agg` off -- its default -- the plan is a plain
        Seq Scan and the query takes 1407 ms; with it on the plan is
        `Custom Scan (theodb_columnar_agg)` and it takes 108 ms. Thirteen times,
        decided by a GUC, with the catalog reporting a columnar table either way.
        """
        marker = type(self).ANALYTICAL_PLAN_MARKERS.get(table.path)
        if marker is None:
            return
        # Per query, not once per table: pushdown coverage depends on the query
        # shape. Measured on the built image at one million rows, with the
        # pushdown on, `sum(amount)` plans as Custom Scan (theodb_columnar_agg)
        # while `GROUP BY category` falls back to Seq Scan -> external-merge Sort
        # (25 456 kB spilled) -> GroupAggregate, and runs 14x slower than heap.
        # A gate that probed one query and generalised would call the second one
        # pushed down.
        probe = query if query is not None else AnalyticalQuery(id="filtered_sum", description="")
        sql = self._analytical_query_sql(table, probe)
        plan = self._fetch_one(f"EXPLAIN (COSTS OFF) {sql}")
        plan_text = str(plan[0]) if plan and plan[0] else ""
        if marker not in plan_text:
            raise AdapterError(
                f"{table.name} is stored in the {table.path} path and the plan for "
                f"query {probe.id!r} did not use it: {marker!r} is absent. Residency "
                f"is necessary and not sufficient, and pushdown coverage varies by "
                f"query shape — measured, the same table is 13x slower on a query "
                f"whose plan falls back. Plan: {plan_text[:300]}",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT,
                    system=self.system_id,
                    details={"path": table.path, "marker": marker},
                ),
            )

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

    def _supports_binary_copy(self) -> bool:
        """pgvector's `vector` has a fixed, documented binary layout: int16 dim,
        int16 unused, then big-endian float4. `copy_binary` writes it."""
        return True

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

    def _build_session_settings(self, index: IndexSpec) -> dict[str, str]:
        """Session settings an index needs in force *while it is being built*.

        Distinct from the search knobs: those are applied per operating point,
        after the index exists. A build-time switch applied afterwards changes
        nothing about the index that was already written, so an adapter that
        treated one as the other would build one structure and label it another.
        """
        return {}

    def _apply_build_session(self, index: IndexSpec) -> None:
        mapping = self._build_session_settings(index)
        if not mapping:
            return
        for guc, literal in mapping.items():
            self._execute(f"SET {guc} = {literal}")
        # Verified in force, exactly as the search knobs are.
        self._verified_search_settings(mapping)

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        self.require(index.capability, f"index kind {index.kind!r}")
        if index.kind == "none":
            return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})
        self._apply_build_session(index)

        name, ddl = self.index_ddl(spec, index)
        started = time.perf_counter()
        with self._under_bulk_budget():
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

    #: pgvector's index is only chosen when `ORDER BY` repeats the distance
    #: expression rather than naming the alias, so the probe is bound twice.
    ORDER_REPEATS_DISTANCE: ClassVar[bool] = True


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

    #: Measured on the built image: `pg_am` holds `theodb_columnar` with
    #: `amtype = 't'`, so the columnar surface here is *storage*, not a cache.
    #: `CREATE TABLE ... USING theodb_columnar` puts the rows in it, and
    #: `pg_class.relam` proves they are there -- there is no "enabled but empty"
    #: state to detect, which is the state that makes the competitor's engine
    #: dangerous to measure.
    ANALYTICAL_PATHS: ClassVar[dict[str, str | None]] = {
        "row": None,
        "columnar": "theodb_columnar",
        # Parquet is a file. The rows land in a heap table first and are then
        # written out; the queries read the file back.
        "parquet": None,
    }

    #: `theodb.enable_columnar_agg` ships **off**, and the project's own wiki
    #: records that the columnar's gain lives in the pushdown rather than in a
    #: plain scan. Measured at one million rows: off -> Seq Scan, 1407 ms;
    #: on -> Custom Scan (theodb_columnar_agg), 108 ms. Leaving it at the default
    #: measures columnar storage without its pushdown, which is a path already
    #: known to lose to heap -- and publishing that as "our columnar" would be
    #: the same error as measuring ScaNN with its AH quantizer off.
    ANALYTICAL_SESSION_SETTINGS: ClassVar[dict[str, dict[str, str]]] = {
        "columnar": {"theodb.enable_columnar_agg": "on"}
    }

    ANALYTICAL_PLAN_MARKERS: ClassVar[dict[str, str]] = {"columnar": "theodb_columnar_agg"}

    def _supports_binary_copy(self) -> bool:
        return True

    #: `over_fetch` is the rescore pool of the AQ+AH scan: `customscan.rs` computes
    #: `rerank_pool = 64 * theodb_hnsw.over_fetch`, and that second stage is what
    #: makes this path comparable to AlloyDB's `pre_reordering_num_neighbors`.
    #: Without it declared, the harness can only sweep probe depth and the two
    #: rescore pools stay at whatever each engine defaults to -- which is how a
    #: quantized index gets measured at its quantizer's fidelity instead of at its
    #: real operating point.
    SEARCH_PARAMETERS: ClassVar[frozenset[str]] = frozenset({"ef_search", "probes", "over_fetch"})

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
            # Reached as of the analytical surface: `USING theodb_columnar` plus a
            # `pg_class.relam` proof.
            "columnar": True,
            # `write_parquet` on load, `read_parquet` in every query.
            "parquet": True,
            # `bm25_build` on load, `bm25_search` per query.
            "lexical": True,
            # `theodb.graph_build` folds the CSR, `graph_expand` walks it, and
            # `graph_expand_card` reports the work done.
            "graph": True,
            # `ai.hybrid_search_rrf` fuses both legs inside the engine.
            "hybrid": True,
            # `pq_subspaces`, `sbq_bits` and `rabitq_bits` are real reloptions, and
            # `vector/sift/pg-scann` builds with `pq_subspaces=64`.
            "vector_quantized": True,
            # `rerank`, `vectorizer` and `ai_sql` stay out: each reaches an
            # external model, and without an endpoint there is nothing to measure.
            # A stub would put a number where an absence belongs.
        }

    def _parquet_path(self, table: AnalyticalTable) -> str:
        """Where the Parquet file for a table lives.

        Configurable rather than fixed, because the writer is the **server**
        process: the directory has to be one the database user can write, which is
        a property of the deployment and not of the harness. Measured the hard way
        -- a directory created by root inside the container gave
        `Permission denied (os error 13)` from `write_parquet`.
        """
        return f"{self.config.parquet_directory.rstrip('/')}/{table.name}.parquet"

    def _after_analytical_load(self, table: AnalyticalTable) -> None:
        """Write the Parquet file the queries will read back.

        The rows are loaded into an ordinary table first because `write_parquet`
        takes a relation. What is measured afterwards is the file: the queries
        read through `read_parquet`, so the heap table is scaffolding rather than
        the path under test.
        """
        if table.path != "parquet":
            return
        self._execute(
            f"SELECT write_parquet({_literal(table.name)}, {_literal(self._parquet_path(table))})"
        )

    def _analytical_query_sql(self, table: AnalyticalTable, query: AnalyticalQuery) -> str:
        if table.path != "parquet":
            return super()._analytical_query_sql(table, query)
        # `read_parquet` returns SETOF jsonb, so the columns are projected out of
        # the document rather than named directly.
        source = (
            f"(SELECT (doc->>'id')::int AS id, (doc->>'amount')::double precision AS amount, "
            f"doc->>'category' AS category, (doc->>'quantity')::int AS quantity "
            f"FROM read_parquet({_literal(self._parquet_path(table))}) AS doc) AS parquet_rows"
        )
        template = type(self).ANALYTICAL_SQL.get(query.id)
        if template is None:
            raise AdapterError(
                f"unknown analytical query {query.id!r}",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT, system=self.system_id, details={"query": query.id}
                ),
            )
        return template.replace("{table}", source)

    # ------------------------------------------------------------- lexical

    def load_documents(self, spec: DocumentTableSpec, documents: Sequence[Document]) -> LoadOutcome:
        """Load documents and build the BM25 index over them.

        The index is built here rather than lazily at search time because
        `bm25_search` over an index that was never built used to return zero rows,
        which is indistinguishable from nothing matching. The engine now refuses
        that outright; loading and building together means the harness never
        depends on which behaviour it gets.
        """
        table = _identifier(spec.table)
        text_column = _identifier(spec.text_column)
        vector_column = _identifier(spec.embedding_column)
        tsv_column = _identifier(f"{spec.text_column}_tsv")
        started = time.perf_counter()
        self._execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        # Both legs in one table: the lexical index reads the text and a dense or
        # hybrid leg reads the vector. Loading only the text would make the
        # hybrid surface unreachable from the same corpus, and comparing two legs
        # over different corpora compares the corpora.
        # The tsvector is generated, not chosen. Measured on the shipped image:
        # `ai.hybrid_search_rrf` defaults to `lexical_engine='ts_rank_cd'`, which
        # calls `ts_rank_cd(tsvector, tsquery)`; and its `'bm25'` engine refuses
        # outright — "requires the pg_textsearch extension ... not present on the
        # shipped image". So the fusable lexical leg is ts_rank_cd, and it needs a
        # tsvector column to exist at all. A generated column keeps it derived from
        # the text rather than maintained separately.
        self._execute(
            f"CREATE TABLE {table} (id bigint PRIMARY KEY, {text_column} text NOT NULL, "
            f"{vector_column} {self.column_type(spec.dimension)} NOT NULL, "
            f"{tsv_column} tsvector GENERATED ALWAYS AS "
            f"(to_tsvector('english', {text_column})) STORED)"
        )
        with (
            self._cursor() as cursor,
            cursor.copy(f"COPY {table} (id, {text_column}, {vector_column}) FROM STDIN") as copy,
        ):
            for document in documents:
                copy.write_row((document.id, document.text, self._to_column(document.vector)))
        self._execute(f"ANALYZE {table}")

        built = self._fetch_one(
            "SELECT bm25_build(%s, %s, %s, %s)",
            (self._lexical_index_id(spec), spec.table, "id", spec.text_column),
        )
        self._lexical_built.add(spec.table)
        counted = self._fetch_one(f"SELECT count(*) FROM {table}")
        loaded = int(counted[0]) if counted else 0
        _ = built
        return LoadOutcome(
            seconds=time.perf_counter() - started,
            rows_loaded=loaded,
            rows_expected=len(documents),
        )

    def _lexical_index_id(self, spec: DocumentTableSpec) -> int:
        """A stable id per table, since bm25_build is keyed by integer."""
        return abs(hash(spec.table)) % 1_000_000

    def execute_lexical(self, query: LexicalQuery) -> RankedResult:
        if query.table not in self._lexical_built:
            raise UnsupportedCapabilityError(
                f"the BM25 index over {query.table} was never built in this session, "
                f"and searching one that does not exist returns zero rows -- "
                f"indistinguishable from nothing matching",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        spec_id = abs(hash(query.table)) % 1_000_000
        started = time.perf_counter()
        rows = self._fetch_all(
            "SELECT id, score FROM bm25_search(%s, %s, %s)",
            (spec_id, query.text, int(query.n)),
        )
        elapsed = time.perf_counter() - started
        return RankedResult(
            ids=tuple(int(row[0]) for row in rows),
            scores=tuple(float(row[1]) for row in rows),
            latency_seconds=elapsed,
        )

    def execute_hybrid(self, query: HybridQuery) -> RankedResult:
        """Fuse both legs with the engine's own RRF.

        The benchmark fuses the legs offline as well, so the engine's fusion can be
        compared to a reference rather than trusted -- which is the point of having
        `analysis/fusion.py` at all.

        The lexical leg here is **ts_rank_cd, not BM25**, and that is a property of
        the shipped image rather than a choice. Measured: `lexical_engine='bm25'`
        refuses with "requires the pg_textsearch extension ... not present on the
        shipped image". So a hybrid number from this image does not exercise the
        BM25 index that `load_documents` builds, and a report must not imply it
        does.
        """
        if query.table not in self._lexical_built:
            raise UnsupportedCapabilityError(
                f"the lexical leg over {query.table} was never built, and fusing one "
                f"leg with nothing returns the dense ranking under a hybrid label",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        started = time.perf_counter()
        rows = self._fetch_all(
            "SELECT id, score FROM ai.hybrid_search_rrf("
            "tbl => %s::regclass, id_col => 'id', content_tsv_col => %s, "
            "vector_col => %s, query_text => %s, query_vector => %s::vector, "
            "result_limit => %s, content_text_col => %s)",
            (
                query.table,
                # The generated tsvector, which is what ts_rank_cd needs.
                "content_tsv",
                "embedding",
                query.text,
                self._to_column(query.vector),
                int(query.n),
                "content",
            ),
        )
        elapsed = time.perf_counter() - started
        return RankedResult(
            ids=tuple(int(row[0]) for row in rows),
            scores=tuple(float(row[1]) for row in rows),
            latency_seconds=elapsed,
        )

    # --------------------------------------------------------------- graph

    def load_graph(
        self, spec: GraphSpec, edges: Sequence[tuple[int, int]], vertex_count: int
    ) -> BuildOutcome:
        """Load the edge list and fold it into the persisted CSR.

        The fold is part of loading rather than of traversing, for the same reason
        the BM25 index is: `graph_expand` over a relation with no CSR answers with
        an empty set, and an empty neighbourhood is a legitimate answer for an
        isolated vertex. The two are indistinguishable after the fact.
        """
        table = _identifier(spec.name)
        started = time.perf_counter()
        self._execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        self._execute(f"CREATE TABLE {table} (src bigint NOT NULL, dst bigint NOT NULL)")
        with (
            self._cursor() as cursor,
            cursor.copy(f"COPY {table} (src, dst) FROM STDIN") as copy,
        ):
            for source, target in edges:
                copy.write_row((source, target))
        self._execute(f"ANALYZE {table}")
        self._execute(f"SELECT theodb.graph_build({_literal(spec.name)}, 'src', 'dst')")
        self._graphs_built.add(spec.name)

        # Timed as a build, per the contract: folding a CSR is structure work,
        # and charging it to a query would make every traversal look expensive.
        size = self._scalar(f"SELECT pg_relation_size({_literal(spec.name)})")
        return BuildOutcome(
            seconds=time.perf_counter() - started,
            index_size_bytes=int(size) if size is not None else None,
            parameters_in_force={"edges": len(edges), "vertices": vertex_count},
        )

    def traverse(self, query: TraversalQuery) -> TraversalResult:
        if query.graph not in self._graphs_built:
            raise UnsupportedCapabilityError(
                f"the CSR for {query.graph} was never built in this session, and "
                f"expanding a graph that has none returns an empty set -- which is "
                f"also what an isolated vertex returns",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        started = time.perf_counter()
        rows = self._fetch_all(
            "SELECT theodb.graph_expand(%s, %s, %s)",
            (query.graph, [int(query.source)], int(query.hops)),
        )
        elapsed = time.perf_counter() - started
        # The work done, asked of the engine rather than inferred from the answer:
        # a traversal returning few vertices after walking many edges is expensive,
        # and the answer size alone would hide that.
        card = self._fetch_one(
            "SELECT theodb.graph_expand_card(%s, %s, %s)",
            (query.graph, [int(query.source)], int(query.hops)),
        )
        return TraversalResult(
            vertices=tuple(int(row[0]) for row in rows),
            edges_visited=int(card[0]) if card and card[0] is not None else 0,
            latency_seconds=elapsed,
        )

    def graph_stats(self) -> dict[str, Any]:
        """Structure size, for bytes-per-edge accounting."""
        return {
            "graphs_built": sorted(self._graphs_built),
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
            elif name == "over_fetch":
                mapping["theodb_hnsw.over_fetch"] = str(int(value))
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
