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

    def fetch_one(self, sql: str, parameters: tuple[object, ...] | None = None):
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
    that does not exist."""
    caps = AlloyDBOmniAdapter().capabilities()

    assert caps["vector_scann"] is True
    assert caps["vector_exact"] is True
    for absent in ("disaggregated_storage", "managed_failover", "read_pool", "columnar"):
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
