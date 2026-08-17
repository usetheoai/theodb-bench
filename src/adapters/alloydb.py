"""AlloyDB Omni: PostgreSQL with Google's ScaNN index and columnar engine.

Omni is a **query layer** -- planner extensions, the `scann` access method and a
columnar engine on top of ordinary PostgreSQL storage. It is not the managed
AlloyDB service: there is no disaggregated storage, no read pool and no managed
failover here, and `capabilities()` says so by omission rather than by claiming
them.

Everything below was measured against `google/alloydbomni:latest` on an
ephemeral droplet on 2026-08-17 rather than read from documentation:

  * the image serves **PostgreSQL 17.9** while TheoDB is PostgreSQL 18, so a
    head-to-head crosses a major version. `export_config` reads the version off
    the server for exactly this reason -- the image tag says `latest` and means
    nothing;
  * `alloydb_scann` is **not** installed by default, and pulls in `vector`
    (Google's fork of pgvector 0.8.2) through CASCADE. That fork is why this
    adapter inherits PgvectorAdapter: the `vector` type, the `<=>`/`<->`/`<#>`
    operators and the text input format are the same;
  * the `scann` access method names its operator classes `cosine`,
    `dot_product` and `l2` -- none of pgvector's `vector_*_ops` names;
  * `SET scann.num_leaves_to_search = 500` **succeeds** in a session that never
    ran `LOAD 'alloydb_scann'`, `current_setting` echoes the value back, and
    `pg_settings` does not list the GUC. The engine keeps searching at its
    default of 0. `shared_preload_libraries` does not carry the library, so the
    LOAD is needed once per session.

That last point is the whole reason the knob gate landed before this adapter:
the gate reads `pg_settings`, so a session that lost its LOAD is refused
instead of measured.
"""

from __future__ import annotations

from typing import Any, ClassVar

from theodb_bench.adapters.base import AnalyticalQuery, AnalyticalTable, IndexSpec
from theodb_bench.adapters.postgres import PgvectorAdapter, _identifier, _literal
from theodb_bench.errors import AdapterError, ErrorContext, Phase


class AlloyDBOmniAdapter(PgvectorAdapter):
    """AlloyDB Omni, exercised through its `scann` access method."""

    system_id = "alloydbomni"
    extension = "alloydb_scann"

    #: The library that has to be in the session for `scann.*` to be real GUCs.
    #: Declared rather than loaded by an override: the base class issues the LOAD
    #: for every adapter that names one, so a third engine cannot forget it.
    library: ClassVar[str | None] = "alloydb_scann"

    #: Measured from `pg_opclass` on the running server. The `scann` names are
    #: the engine's own; the pgvector rows come from the bundled fork and are
    #: kept so an Omni-vs-Omni comparison of scann against hnsw is possible.
    OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {
        "scann": {"l2": "l2", "ip": "dot_product", "cosine": "cosine"},
        "hnsw": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
        "ivfflat": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
    }

    #: The scann knobs this adapter applies. `ef_search` is deliberately absent:
    #: measured on the running server, Omni's bundled pgvector fork registers no
    #: `hnsw.*` GUC at all, and recall was identical at ef_search 16 and 256
    #: (0.7820 both). Declaring it would let a sweep publish one operating point
    #: under three labels.
    SEARCH_PARAMETERS: ClassVar[frozenset[str]] = frozenset(
        {
            "num_leaves_to_search",
            "pct_leaves_to_search",
            "pre_reordering_num_neighbors",
        }
    )

    def capabilities(self) -> dict[str, bool]:
        """What this adapter can exercise -- not what AlloyDB can do.

        Omni also ships `google_columnar_engine` (installed and preloaded by
        default, measured) and `google_ml_integration`. This adapter reaches
        neither: the analytical and AI lifecycle methods are not implemented
        here, and a capability is a statement about this code path.

        The platform features of managed AlloyDB -- disaggregated storage, read
        pools, managed failover -- are absent from this map because Omni does
        not have them at all. Declaring them would make a race measure a
        product that is not running.
        """
        return {
            "vector_exact": True,
            "vector_scann": True,
            "vector_hnsw": True,
            "vector_ivfflat": True,
            "vector_filtered": True,
            # Reached as of the analytical surface. Declared only because the
            # residency gate below refuses every state in which the label would
            # be a lie -- without that gate this would be the false claim the
            # docstring above warns about.
            "columnar": True,
        }

    def _build_session_settings(self, index: IndexSpec) -> dict[str, str]:
        """AH quantization is enabled at build time or it is not enabled at all.

        Measured on the running server: `CREATE INDEX ... WITH (quantizer='AH')`
        fails with `AH quantization is not enabled for the index` unless
        `scann.enable_ah_quantizer` is on in the session doing the build. With it
        on, the reloption records `quantizer=AH`.

        This matters more than a flag usually does. AH is the anisotropic
        quantizer ADR-0035 credits for the ~25x QPS gap it measured against the
        ScaNN library, and it ships **off** -- so a run that left the default
        would build SQ8, measure scalar quantization, and answer a question about
        AH with it.
        """
        if str(index.parameters.get("quantizer", "")).upper() == "AH":
            return {"scann.enable_ah_quantizer": "on"}
        return {}

    def _search_guc_mapping(self, parameters: dict[str, Any]) -> dict[str, str]:
        """The scann search knobs, by the names `pg_settings` uses.

        Only the knobs a benchmark actually sweeps are mapped. `num_leaves` is a
        build-time reloption, not a search GUC, and is not accepted here so a
        typo cannot pass as a search parameter.
        """
        mapping: dict[str, str] = {}
        for name, value in parameters.items():
            if name == "num_leaves_to_search":
                mapping["scann.num_leaves_to_search"] = str(int(value))
            elif name == "pct_leaves_to_search":
                mapping["scann.pct_leaves_to_search"] = str(int(value))
            elif name == "pre_reordering_num_neighbors":
                # How many quantized candidates get rescored with exact
                # distances. Ships at -1, and that default caps recall at the
                # quantizer's fidelity: measured at 100k SIFT-128 with AH and 80
                # leaves searched, recall goes from 0.6568 at the default to
                # 0.9964 at 100 and 0.9998 at 500. Searching four times more
                # leaves had bought 1.4 points, so the ceiling was quantization
                # error and not search depth. A frontier measured at the default
                # would report the competitor topping out at two thirds recall,
                # which is false and happens to flatter us.
                mapping["scann.pre_reordering_num_neighbors"] = str(int(value))
        return mapping

    # ----------------------------------------------------------- analytical
    #
    # The columnar engine is a cache populated by policy, not storage. Measured on
    # the running server, it has four distinguishable states, and three of them
    # answer queries correctly while silently falling back to heap:
    #
    #   1. `enabled = off` (the default). Reading g_columnar_columns errors.
    #   2. enabled, never populated: 0 columns, 0 MB, Seq Scan.
    #   3. enabled, `google_columnar_engine_add()` called, container /dev/shm at
    #      Docker's 64 MB default: g_columnar_columns = 4 while the engine summary
    #      reports Memory Used = 0 MB, and the plan is still a Seq Scan. The
    #      refresh fails with `could not resize shared memory segment ... No space
    #      left on device`.
    #   4. enabled, populated, --shm-size=4g: Memory Used = 42 MB and the plan
    #      carries `Parallel Custom Scan (columnar scan)`.
    #
    # State 3 is why this gate is not built on g_columnar_columns, which the
    # published independent evaluation recommends as the residency proof: the view
    # reports *registration*. Residency is `Memory Used > 0`, and even that is not
    # sufficient -- at 50 000 rows the store was loaded and the planner still
    # chose a sequential scan.

    #: The columnar path is a heap table plus a cache registration, so no access
    #: method is named for it. `_after_analytical_load` does the registration.
    ANALYTICAL_PATHS: ClassVar[dict[str, str | None]] = {"row": None, "columnar": None}

    def _after_analytical_load(self, table: AnalyticalTable) -> None:
        """Register the table with the columnar engine.

        Auto-columnarization is driven by query history, so a freshly loaded table
        in a freshly started engine is recommended by nothing and stays out of the
        store. Registering explicitly is what makes the run measure the engine
        rather than the heap it silently falls back to.
        """
        if table.path != "columnar":
            return
        self._execute(f"SELECT google_columnar_engine_add({_literal(table.name)})")

    def assert_analytical_path(
        self, table: AnalyticalTable, query: AnalyticalQuery | None = None
    ) -> None:
        """Prove the columnar engine is on, loaded, and actually in the plan.

        Three separate facts, because the three failures need three different
        actions: turn the engine on (and restart), populate the store (and give
        the container shared memory), or accept that at this size the planner
        prefers heap. Collapsing them into one message would send a reader to fix
        the wrong thing.
        """
        if table.path != "columnar":
            super().assert_analytical_path(table, query)
            return

        enabled = self._fetch_one(
            "SELECT setting, context FROM pg_settings WHERE name = %s",
            ("google_columnar_engine.enabled",),
        )
        if enabled is None or str(enabled[0]) != "on":
            raise AdapterError(
                "google_columnar_engine.enabled is not on, so every columnar query "
                "falls back to heap. Its context is `postmaster`: it cannot be SET "
                "for a session, it needs ALTER SYSTEM and a server restart.",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )

        registered = self._fetch_one(
            "SELECT count(*) FROM g_columnar_columns WHERE relation_name = %s",
            (table.name,),
        )
        count = int(registered[0]) if registered and registered[0] is not None else 0
        if count == 0:
            raise AdapterError(
                f"the columnar engine holds no column of {table.name}, so the scan "
                f"would run on heap under the columnar label.",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )

        summary = self._fetch_one(
            "SELECT value FROM g_columnar_engine_summary WHERE name = %s",
            ("Memory Used (MB)",),
        )
        used = int(summary[0]) if summary and summary[0] is not None else 0
        if used <= 0:
            raise AdapterError(
                f"{count} column(s) of {table.name} are registered with the columnar "
                f"engine and the store is not loaded: Memory Used is {used} MB. "
                f"g_columnar_columns reports registration, not residency. Measured "
                f"cause: the refresh needs shared memory the container does not "
                f"have -- run it with --shm-size and check for `could not resize "
                f"shared memory segment`.",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )

        # Probed with the query about to run, when there is one: pushdown coverage
        # depends on the query shape here as much as on ours, and proving it once
        # per table would call an unsupported shape supported.
        sql = (
            self._analytical_query_sql(table, query)
            if query is not None
            else f"SELECT count(*) FROM {_identifier(table.name)}"  # noqa: S608
        )
        plan = self._fetch_one(f"EXPLAIN (COSTS OFF) {sql}")
        plan_text = str(plan[0]) if plan and plan[0] else ""
        if "columnar scan" not in plan_text:
            raise AdapterError(
                f"the store is loaded ({used} MB) and the planner did not use the "
                f"columnar scan for {table.name}: residency is necessary and not "
                f"sufficient. Measured at 50 000 rows with the store loaded, the "
                f"planner still chose a sequential scan. Plan: {plan_text[:300]}",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
