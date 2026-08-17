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

from theodb_bench.adapters.postgres import PgvectorAdapter, _literal


class AlloyDBOmniAdapter(PgvectorAdapter):
    """AlloyDB Omni, exercised through its `scann` access method."""

    system_id = "alloydbomni"
    extension = "alloydb_scann"

    #: The library that has to be in the session for `scann.*` to be real GUCs.
    library = "alloydb_scann"

    #: Measured from `pg_opclass` on the running server. The `scann` names are
    #: the engine's own; the pgvector rows come from the bundled fork and are
    #: kept so an Omni-vs-Omni comparison of scann against hnsw is possible.
    OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {
        "scann": {"l2": "l2", "ip": "dot_product", "cosine": "cosine"},
        "hnsw": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
        "ivfflat": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
    }

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
        }

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        """Wait, create the extension, then load the library into the session.

        The LOAD is not redundant with CREATE EXTENSION: the extension registers
        the access method in the catalog, the LOAD registers the `scann.*` GUCs
        in *this backend*. Without it every `SET scann.…` is a placeholder that
        the server accepts and ignores.
        """
        super().wait_ready(timeout_seconds)
        self._execute(f"LOAD {_literal(self.library)}")

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
            elif name == "enable_ah_quantizer":
                # Measured `off` by default -- the anisotropic quantizer that
                # gives ScaNN its published throughput is opt-in, and a run at
                # the default measures ScaNN without it.
                mapping["scann.enable_ah_quantizer"] = "on" if value else "off"
        return mapping

    def export_config(self) -> dict[str, Any]:
        """Server-reported configuration, with the version read, never inferred.

        The independent AlloyDB evaluation of 2026-08 measured the Docker Hub
        image sitting on PostgreSQL 17 while the Linux packages had moved to 18.
        A version taken from the image tag would have hidden that; a version
        taken from the server cannot.

        A server that does not answer gets no version invented for it: the field
        is simply absent, which a reader can act on.
        """
        payload = super().export_config()
        parts: list[str] = []

        server = self._fetch_one("SELECT version()")
        if server is not None and server[0]:
            parts.append(str(server[0]).split(" on ")[0])

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
