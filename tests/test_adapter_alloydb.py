"""The AlloyDB Omni adapter: what can be proven without a server.

Every expectation here was measured against `google/alloydbomni:latest` on an
ephemeral droplet (138.197.22.192, 2026-08-17) before it was written down. The
measurements that shaped this file:

  * the server is PostgreSQL 17.9 -- a major behind TheoDB, so the version has
    to reach the bundle read from the server rather than inferred from a tag;
  * the scann access method names its operator classes `cosine` /
    `dot_product` / `l2`, not pgvector's `vector_*_ops`;
  * `SET scann.num_leaves_to_search = 500` in a session that has not run
    `LOAD 'alloydb_scann'` SUCCEEDS, `current_setting` echoes 500 back, and
    `pg_settings` does not list the GUC at all. The engine searches at its
    default. `shared_preload_libraries` does not include the library, so the
    LOAD is required per session.

The last one is why this adapter exists downstream of the knob gate: without
the gate, the first race against ScaNN would publish a shallow default as a
result.
"""

from __future__ import annotations

import pytest
from theodb_bench.adapters.alloydb import AlloyDBOmniAdapter
from theodb_bench.adapters.base import IndexSpec, VectorTableSpec
from theodb_bench.errors import AdapterError


class _OmniStub:
    """A server that registers `scann.*` only after it has seen a LOAD.

    This is the measured shape, not an invented one: before the LOAD the GUC is
    absent from `pg_settings`; after it, 111 `scann.*` entries appear and a SET
    records as `source=session`.
    """

    def __init__(self, *, registers_only_after_load: bool = True) -> None:
        self.executed: list[str] = []
        self._loaded = not registers_only_after_load
        self._set: dict[str, str] = {}
        self.server_version = "PostgreSQL 17.9 on x86_64-pc-linux-gnu"
        self.extension_version = "0.1.4"

    def execute(self, sql: str, parameters: tuple[object, ...] | None = None) -> None:
        self.executed.append(sql)
        if sql.startswith("LOAD "):
            self._loaded = True
        elif sql.startswith("SET ") and "=" in sql:
            name, _, value = sql[4:].partition("=")
            self._set[name.strip()] = value.strip()

    def fetch_one(
        self, sql: str, parameters: tuple[object, ...] | None = None
    ) -> tuple[object, ...] | None:
        if "pg_settings" in sql:
            name = str(parameters[0]) if parameters else ""
            if name.startswith("scann.") and not self._loaded:
                return None  # the placeholder case, measured
            value = self._set.get(name)
            return None if value is None else (value, "session")
        if "version()" in sql:
            return (self.server_version,)
        if "pg_extension" in sql:
            return (self.extension_version,)
        return None


def _adapter_with(server: _OmniStub) -> AlloyDBOmniAdapter:
    adapter = AlloyDBOmniAdapter()
    adapter._row_count = 10_000
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]
    return adapter


# ------------------------------------------------------------------ identity


def test_the_scann_opclasses_are_the_names_the_engine_uses() -> None:
    adapter = AlloyDBOmniAdapter()

    assert adapter.opclass("scann", "cosine") == "cosine"
    assert adapter.opclass("scann", "ip") == "dot_product"
    assert adapter.opclass("scann", "l2") == "l2"


def test_the_pgvector_access_methods_are_still_reachable() -> None:
    """Omni ships a fork of pgvector 0.8.2, so hnsw and ivfflat exist here too."""
    adapter = AlloyDBOmniAdapter()

    assert adapter.opclass("hnsw", "cosine") == "vector_cosine_ops"


def test_capabilities_declares_only_what_this_code_exercises() -> None:
    """Omni is a query layer. Claiming platform features would race a product
    that does not exist.

    `columnar` is declared as of the analytical surface, and only because the
    residency gate refuses every state in which the label would be a lie — engine
    off, nothing registered, registered with an empty store, or a plan that never
    reaches the columnar scan. Without that gate this would be exactly the false
    claim the module docstring warns about.
    """
    caps = AlloyDBOmniAdapter().capabilities()

    assert caps["vector_scann"] is True
    assert caps["vector_exact"] is True
    assert caps["columnar"] is True
    for absent in ("disaggregated_storage", "managed_failover", "read_pool", "parquet"):
        assert absent not in caps


# ------------------------------------------------------------------- session


def test_wait_ready_creates_the_extension_and_loads_the_library() -> None:
    server = _OmniStub()
    adapter = _adapter_with(server)

    adapter.wait_ready()

    statements = " | ".join(server.executed)
    assert 'CREATE EXTENSION IF NOT EXISTS "alloydb_scann" CASCADE' in statements
    assert "LOAD 'alloydb_scann'" in statements


def test_the_search_knob_is_the_scann_guc() -> None:
    server = _OmniStub()
    adapter = _adapter_with(server)
    adapter.wait_ready()

    adapter.set_search_parameters({"num_leaves_to_search": 500})

    assert adapter.effective_search_parameters() == {"scann.num_leaves_to_search": "500"}


def test_without_the_load_the_gate_refuses_the_run() -> None:
    """The measured trap: the SET succeeds and nothing is in force.

    Removing the LOAD must not produce a run at a shallow default -- it must
    produce a refusal. This is the knob gate doing its job on a competitor's
    product.
    """
    server = _OmniStub()
    adapter = _adapter_with(server)
    # deliberately skipping wait_ready: no LOAD was issued

    with pytest.raises(AdapterError):
        adapter.set_search_parameters({"num_leaves_to_search": 500})


# ------------------------------------------------------------------- version


def test_the_version_comes_from_the_server_not_from_the_image_tag() -> None:
    server = _OmniStub()
    adapter = _adapter_with(server)

    version = adapter.export_config()["version"]

    assert "17.9" in version
    assert "0.1.4" in version
    assert "latest" not in version


def test_a_server_that_will_not_say_its_version_does_not_get_one_invented() -> None:
    server = _OmniStub()
    server.server_version = ""
    adapter = _adapter_with(server)
    adapter._fetch_one = lambda sql, parameters=None: None  # type: ignore[method-assign]

    payload = adapter.export_config()

    assert "17" not in str(payload.get("version", ""))


# ------------------------------- the AH quantizer is a build-time requirement
#
# Measured on the running server:
#
#   CREATE INDEX ... WITH (quantizer='AH')
#     -> ERROR: AH quantization is not enabled for the index
#   SET scann.enable_ah_quantizer = on;  CREATE INDEX ... WITH (quantizer='AH')
#     -> CREATE INDEX, reloptions = {num_leaves=20, quantizer=AH}
#
# Valid quantizer values are SQ8, Flat and AH. AH is the anisotropic quantizer
# ADR-0035 credits for the ~25x QPS gap it measured against the ScaNN library, and
# it needs the GUC on *at build time* -- so mapping the flag as a search parameter,
# applied after the index exists, silently measures SQ8 instead.


def test_an_ah_index_turns_the_quantizer_on_before_building() -> None:
    server = _OmniStub()
    adapter = _adapter_with(server)
    adapter.wait_ready()

    settings = adapter._build_session_settings(
        IndexSpec(kind="scann", parameters={"num_leaves": 100, "quantizer": "AH"})
    )

    assert settings == {"scann.enable_ah_quantizer": "on"}


def test_a_non_ah_index_does_not_turn_it_on() -> None:
    """SQ8 and Flat need nothing, and enabling AH for them would change what ran."""
    adapter = AlloyDBOmniAdapter()

    for quantizer in ("SQ8", "Flat"):
        settings = adapter._build_session_settings(
            IndexSpec(kind="scann", parameters={"quantizer": quantizer})
        )
        assert settings == {}, quantizer


def test_the_build_setting_is_verified_in_force_not_just_sent() -> None:
    """A GUC accepted and not registered would build SQ8 under the AH label.

    The placeholder case, on the build axis this time: the SET succeeds and
    `pg_settings` never lists the GUC, so the index is written with the default
    quantizer while the bundle says AH.
    """

    class _SwallowsTheSet(_OmniStub):
        def execute(self, sql: str, parameters: tuple[object, ...] | None = None) -> None:
            # Accepts everything, registers nothing -- what an unloaded library does.
            self.executed.append(sql)
            if sql.startswith("LOAD "):
                self._loaded = True

    server = _SwallowsTheSet()
    adapter = _adapter_with(server)
    adapter.wait_ready()

    with pytest.raises(AdapterError, match=r"not a registered setting|placeholder"):
        adapter.build_index(
            VectorTableSpec(table="t", dimension=8, metric="l2"),
            IndexSpec(kind="scann", parameters={"quantizer": "AH"}),
        )


def test_the_rerank_depth_is_a_declared_search_knob() -> None:
    """Without it, an AH frontier is a quantization-error frontier.

    Measured at 100k SIFT-128 with quantizer=AH and num_leaves_to_search=80:

        pre_reordering_num_neighbors = -1 (default)  ->  recall@10 = 0.6568
        pre_reordering_num_neighbors = 100           ->  recall@10 = 0.9964
        pre_reordering_num_neighbors = 500           ->  recall@10 = 0.9998

    The 0.66 ceiling was entirely the missing exact-distance rescore, which is how
    ScaNN is designed to work. A frontier measured at the default would have
    reported the competitor topping out at two thirds recall — a false claim, and
    one that happened to flatter us.
    """
    server = _OmniStub()
    adapter = _adapter_with(server)
    adapter.wait_ready()

    adapter.set_search_parameters({"num_leaves_to_search": 80, "pre_reordering_num_neighbors": 100})

    assert adapter.effective_search_parameters() == {
        "scann.num_leaves_to_search": "80",
        "scann.pre_reordering_num_neighbors": "100",
    }
