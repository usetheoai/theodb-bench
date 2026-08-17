# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **SIFT1M is a registered dataset** (B-057): one million 128-dimensional SIFT descriptors with 10 000 queries,
  identified by the checksum of the bytes actually downloaded
  (`dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`, 525 128 288 bytes) rather than by a
  version string. It is the corpus ADR-0035 used when it measured the ~25x QPS gap against the ScaNN
  *library*, so it is the only corpus on which a measurement against the scann *access method* can be
  compared to that conclusion. The licence is recorded as unverified and `redistributable: false`, because
  the TEXMEX/INRIA corpus repackaged by ANN-Benchmarks carries no licence text this project could confirm —
  and inventing one to fill the field is the fabrication the manifest exists to prevent.
- **The ScaNN rerank depth is a declared, swept search knob** (B-057), and without it an AH frontier measures
  quantization error rather than the index. Measured at 100 000 SIFT-128 vectors with `quantizer=AH` and 80 of
  316 leaves searched: `scann.pre_reordering_num_neighbors` ships at `-1`, and at that default recall@10 is
  **0.6568**; at 100 it is **0.9964**; at 500, **0.9998**. Searching four times more leaves had bought 1.4
  points, so the ceiling was the missing exact-distance rescore, which is how ScaNN is designed to work. A
  frontier taken at the default would have reported the competitor topping out at two thirds recall — false,
  and it happened to flatter us.
- **AH quantization is applied at build time, and verified in force** (B-057). Measured:
  `CREATE INDEX ... WITH (quantizer='AH')` fails with `AH quantization is not enabled for the index` unless
  `scann.enable_ah_quantizer` is on in the building session; valid quantizer values are `SQ8`, `Flat` and `AH`,
  and the flag ships off. Build-time settings are therefore a separate declaration from the search knobs: one
  applied after the index exists changes nothing about the index already written, so an adapter treating one as
  the other builds SQ8 and labels it AH.
- **A scann search sweep is a registered suite** (B-057): `vector/synthetic/scann-sweep` sweeps
  `num_leaves_to_search` over a scann index, as the hnsw suite sweeps `ef_search`. They are separate suites
  because the search knob belongs to the index family and one suite cannot ask for both — an adapter that
  cannot apply a requested knob now refuses the run. The two families are compared at **matched recall** from
  their frontiers, never by pairing knob values that mean different things.
- **The analytical surface reaches real engines** (B-061): `load_analytical` and `execute_analytical` are
  implemented for the PostgreSQL family. Before this, only the in-process fake could execute an analytical
  query — the 336-line `AnalyticalBenchmark`, its oracle and its three declared execution paths had no
  engine attached to them. The heap path is a plain table; TheoDB's columnar path is
  `CREATE TABLE ... USING theodb_columnar`; AlloyDB Omni's is a heap table plus a cache registration. Two
  mechanisms under one label, declared per adapter rather than derived from the label.
- **A residency gate that refuses every state in which the columnar label would be a lie** (B-061).
  Measured against a running AlloyDB Omni, its columnar engine has four distinguishable states and three of
  them answer queries correctly while silently falling back to heap: engine off (the default, and its
  context is `postmaster`, so only a restart changes it); enabled but never populated; **enabled and
  registered with an empty store**; and enabled, populated, actually used. The third one is why the gate is
  not built on `g_columnar_columns`, which the published independent evaluation recommends as the residency
  proof: measured, that view reported **4 columns while the engine summary reported Memory Used = 0 MB**,
  and the plan was still a sequential scan. It reports registration, not residency. The measured cause is
  that the refresh needs shared memory a default Docker container does not have — it fails with
  `could not resize shared memory segment`. Each state produces a different message, because they need
  different actions.
- The columnar aggregate pushdown is enabled and **verified in force** before an analytical number is taken
  (B-061). `theodb.enable_columnar_agg` ships off. Measured on the built image, same table of one million
  rows and same query: off → `Seq Scan`, **1407 ms**; on → `Custom Scan (theodb_columnar_agg)`, **108 ms**.
  Thirteen times, decided by a GUC, with the catalog reporting a columnar table either way. Leaving it at
  the default measures columnar storage without its pushdown — a path the project already knows loses to
  heap — and publishing that as "our columnar" would be the same error as measuring ScaNN with its AH
  quantizer off.
- The plan proof is taken **per query, not per table** (B-061), because pushdown coverage depends on the
  query shape. Measured at one million rows with the pushdown on: `sum(amount)` plans as
  `Custom Scan (theodb_columnar_agg)`, while `GROUP BY category` falls back to Seq Scan → external-merge
  Sort (25 456 kB spilled to disk) → GroupAggregate and runs **14× slower than heap**. A gate that probed
  one query and generalised would have called the grouped one pushed down.
- **AlloyDB Omni is a measurable system** (B-059): `alloydbomni` is a registered adapter driving
  Google's `scann` access method. Every property below was measured against
  `google/alloydbomni:latest` on an ephemeral droplet rather than read from documentation.
  `capabilities()` declares only what this code exercises, and deliberately claims none of the
  managed service's platform features — Omni is a query layer, with no disaggregated storage,
  read pool or managed failover. Racing against capabilities the running product does not have
  would measure a product that is not there.
- The adapter issues `LOAD 'alloydb_scann'` per session, and the knob gate from B-060 is what
  proves it took effect (B-059). Measured: in a session without the LOAD,
  `SET scann.num_leaves_to_search = 500` **succeeds**, `current_setting` echoes `500` back, and
  `pg_settings` does not list the GUC — the engine keeps searching at its default of 0.
  `shared_preload_libraries` does not carry the library. This is the same placeholder mechanism
  the gate was built for, now confirmed on another engine; removing the LOAD makes two tests
  fail rather than producing a shallow result.
- The measured server version reaches the bundle read from the server, never inferred from the
  image tag (B-059). The published image serves **PostgreSQL 17.9**, so a head-to-head against
  TheoDB crosses a major version — a fact a report has to state, not hide. A server that will
  not answer gets no version invented for it: the field is omitted.

- **A requested search knob the adapter cannot apply is now refused** (B-059). The gate added in
  B-060 verified every knob it *mapped* and silently accepted every knob it did not — and a second
  engine is what exposed the difference. Measured against a running AlloyDB Omni: its bundled
  pgvector fork registers no `hnsw.*` GUC at all (zero rows in `pg_settings`), and the Omni adapter
  maps `num_leaves_to_search`, not `ef_search`. A sweep of `ef_search` therefore produced an empty
  mapping, the gate had nothing to check, and it passed vacuously: recall measured **0.7820 at both
  ef_search=16 and ef_search=256**, and the bundle published three rows labelled 16 / 64 / 256 that
  were one operating point. Each adapter now declares the knobs it understands, and a request naming
  anything else fails the run instead of relabelling a default. The same command that produced the
  fictional rows now reports `INVALID`.

### Changed

- **The TheoDB adapter now emits TheoDB's own access methods** (B-064). Measured against the image this
  project's own Dockerfile builds — PostgreSQL 18.6, `theodb_rs` 1.5.0 — the harness emitted
  `CREATE INDEX ... USING hnsw ("embedding" vector_l2_ops)` and the server answered
  `access method "hnsw" does not exist`. `pg_am` holds `theodb_hnsw` and `theodb_ivfflat`, with
  `theodb_hnsw_l2_ops` and friends. The bare `hnsw` name and the `vector_*_ops` classes do exist — in the
  separate `vector` compatibility shim, which the image creates in `template1` rather than in the
  `postgres` database a client reaches by default. So every indexed row of our own product's axis
  returned `INVALID`, while the exact-search row measured and was published: the bundle was not empty,
  it was partial. The engine's access-method name is now declared per adapter, while the bundle label
  stays the index family. Verified: the same run is `VALID` with a real recall curve
  (0.5928 → 0.7800 → 0.9650 across ef_search 16 / 64 / 256).
- The TheoDB adapter loads `theodb_rs` into the session (B-064). Measured: a fresh session holds zero
  `theodb%` rows in `pg_settings`, and no `hnsw.ef_search` either, until the LOAD runs — so every swept
  `ef_search` was a placeholder and the search ran at the default of 64. The LOAD is now issued by the
  base class for any adapter that declares a library, so a third engine cannot forget it.
- The server version reaches the bundle alongside the extension version, for every adapter that has an
  extension (B-064). One machine, one afternoon: TheoDB on PostgreSQL 18.6, pgvector on 17.11, AlloyDB
  Omni on 17.9. The comparison crosses a major version, and the only bundle that hid which PostgreSQL it
  ran on was our own product's, because its override replaced the base version instead of composing with
  it. Three separate implementations became one.
- Index parameters are rendered by type instead of forced through `int()` (B-059). Measured:
  `scann` accepts `quantizer='sq8'`, a string, and the previous renderer raised a bare
  `ValueError` with no phase, system or option name. Strings are now quoted and escaped through
  the existing literal helper; a type the renderer does not know is refused with an
  `AdapterError` rather than coerced, because a benchmark definition carrying a list where a
  scalar belongs is broken, and stringifying it would put an unintended index configuration into
  a published measurement.
- Operator classes are declared per adapter instead of by one shared table (B-059). Measured:
  the `scann` access method names its three classes `cosine`, `dot_product` and `l2` — none of
  pgvector's `vector_*_ops`. The lookup reads the table off the concrete class, so an adapter
  cannot inherit the wrong convention by accident.
- The contract test asserting that every adapter reports its effective search parameters is now
  parametrized off the registry instead of a written-out list (B-059). The list version was
  measured passing while `alloydbomni` was already registered and uncovered: a test that
  enumerates what it claims to cover universally excludes every adapter added after it was
  written, and reports green for doing so.
- A search parameter is now **verified in force before anything is measured** (B-060). The
  harness already refused to report a number when the planner ignored the index
  (`assert_index_used`); it did not refuse when the *knob* was ignored — the `SET` was issued
  and nothing read the value back. Measured on PostgreSQL 18: `SET nao.existe = 999` succeeds,
  `current_setting` hands back `999`, and `pg_settings` holds no such row. An unregistered
  namespaced GUC is accepted as a placeholder, so `current_setting` cannot detect it and
  `pg_settings` can. The gate reads `setting` and `source` from `pg_settings` and refuses when
  the GUC is absent, when the value diverges from what was sent, or when `source` is still
  `default`.
- The bundle records the search parameters **in force** alongside those requested, keyed by GUC
  name (B-060). The two are not always equal: `probes` is clamped to the list count, so a
  request of 10000 on a 10k-row table is sent as the clamp — and `points[].parameters` was built
  from the request, before the knobs were applied. No schema version changed: that field is
  already declared as an open object of scalars.
- `SystemAdapter.effective_search_parameters()` is part of the contract, and every registered
  adapter answers it — including `FakeAdapter`, which is the double the runner's own tests
  exercise most, so a contract that skipped it would be untested where it runs most (B-060).
- Versioned JSON schemas for every machine-readable artifact: benchmark,
  manifest, environment, dataset, system, validation, result, statistics,
  regression, pareto and summary. Artifacts are validated before being written,
  so an invalid file never lands in a bundle.
- `theodb-bench doctor`: fifteen host checks reporting PASS, WARN, FAIL or
  UNAVAILABLE. Which checks are mandatory depends on the profile, so a laptop
  can run a smoke benchmark and cannot produce a release claim.
- `theodb-bench env`: full environment capture from procfs and sysfs, with
  every undeterminable field recorded as an explicit absence carrying its
  reason.
- Immutable run bundles: finalization freezes the manifest and every raw
  measurement, while still allowing re-analysis to add new derived artifacts.
- Resource isolation with escape detection: a subprocess that leaves the
  declared CPU allocation is caught even on hosts where nothing could be
  enforced.
- Telemetry collectors (process, perf) that can be switched off and that
  measure their own overhead. A counter that could not be collected is recorded
  as absent, never as zero.
- Dataset layer identifying datasets by checksum: `dataset list`, `verify` and
  `fetch`, with atomic download and refusal to silently replace mismatched
  bytes.
- System adapter contract plus four adapters: a deterministic fake that
  produces nine real failure modes on demand, upstream PostgreSQL, pgvector and
  TheoDB.
- Vector ANN workload with untimed warm-up, per-configuration index isolation,
  query caps that appear in the label, and recall computed by the benchmark
  from its own oracle.
- Eleven-phase run orchestrator producing a complete, validated, immutable
  bundle.
- Analysis: recall by distance threshold following ANN-Benchmarks, nDCG, MRR,
  recall@n, latency percentiles, best-of-N throughput, aggregation that keeps
  every repetition, stability detection, Pareto frontiers and matched-quality
  selection.
- ANN dataset readers for ANN-Benchmarks HDF5 and the fvecs/ivecs family, and
  `theodb-bench run --dataset` to measure a verified corpus. Published
  distances are never read; recall recomputes them from the vectors.
- Reciprocal rank fusion, as an offline twin of the system's own fusion so the
  two can be compared rather than one trusted.
- Retrieval suite: lexical, dense, hybrid RRF and hybrid plus rerank over one
  corpus and one query set, reporting nDCG@10, Recall@k and MRR alongside
  throughput, with model latency in its own stage.
- Model endpoint abstraction (mock, local, remote) where only the deterministic
  mock may back a regression gate, and the mock's latency is required to be
  non-zero because an instant model changes the loop's concurrency regime.
- Operations suite measuring the foreground write clock and the
  time-to-freshness clock separately, across insert, update, backlog drain and
  worker saturation.
- Graph suite: 1/2/3-hop, BFS, fanout sweep, build and rebuild, with every
  traversal validated against an oracle before its timing is accepted.
- Analytical suite comparing row, columnar and Parquet execution on identical
  data, with per-stage timings and answer validation.
- Paired significance testing: randomisation test, bootstrap confidence
  interval and t-test cross-check, with Monte-Carlo correction and a fixed
  seed. Comparative significance claims are now possible rather than
  forbidden.
- Regression comparison that fails closed on an incomparable baseline and
  reports ADVISORY for any threshold not derived from a measured noise floor.
- Reports in both halves: a human report that leads with status and profile,
  and a machine summary carrying provenance and limitations.
- CI in two classes: shared correctness CI whose numbers are explicitly
  discarded, and a dedicated benchmark workflow that never triggers on a pull
  request.
- Methodology documents covering the measurement-integrity invariants and the
  agent workload surface.
- Agent workload is now the primary benchmark surface; the seven capability
  surfaces are components that explain an agent result rather than substitutes
  for it.
- Dataset manifests are JSON rather than YAML
  (`docs/decisions/0002-json-dataset-manifests.md`).

### Fixed

- **The recall oracle can no longer be OOM-killed by the corpus it is supposed to measure** (B-057).
  `brute_force_ground_truth` was measured being killed at **10.5 GB** of resident memory while building ground
  truth for a 512 MB corpus — one million 128-dimensional vectors against 500 queries, on a 16 GB host. The
  cause was two allocations, neither algorithmic: a `(queries x corpus)` float64 distance matrix (4 GB) and an
  identically shaped int64 tile of `arange(corpus_size)` (4 GB) whose only purpose was breaking ties by id. It
  also full-sorted a million distances per query to take the top ten. It now chunks over queries, keeping the
  working matrix in the tens of megabytes whatever the corpus size, and selects with a partition plus a
  tie-safe re-admission of everything level with the k-th element. Behaviour is unchanged and pinned by tests
  that compare against the previous implementation across all three metrics, including the case a careless
  partition gets wrong: ties spanning the top-k boundary must still resolve by ascending id, or recall stops
  being reproducible. A benchmark harness that cannot build ground truth at the scale it is meant to measure
  is not measuring that scale, and the tests assert the memory ceiling rather than a wall time.

- The module docstring of `src/adapters/postgres.py` no longer claims an invariant the code does not
  enforce. It advertised, as I5, that "the index is forced *and* verified"; measured 2026-08-17,
  `assert_index_used` has no caller anywhere in the package, raises `ProgrammingError` if called
  (this class overrides `_query_sql` to repeat the distance expression, so the probe binds twice
  while the inherited verifier binds once), and `SET enable_seqscan = off` appears in that docstring
  and nowhere else in executable code. The harness measures whatever plan the planner picks. No
  published number is retracted: at the registered suite's size (10 000 × 64) EXPLAIN confirms the
  planner does choose the index on pgvector, Omni/hnsw and Omni/scann — but at 200 rows it chose a
  sequential scan, so the hole is latent rather than harmless. The mechanism is tracked separately;
  what changed here is that the file stops asserting something untrue.
- A run manifest could name a dataset the run never measured: `dataset_id` was
  recorded while the workload generated a synthetic corpus. Declaring a dataset
  now requires supplying the vectors, and supplying vectors requires declaring
  their identity.
- The TheoDB adapter declared hybrid, lexical, columnar, Parquet, graph and
  vectorizer capabilities it does not implement, putting false claims into
  every `system.json`. It now declares only the vector surface it can exercise.

[Unreleased]: https://github.com/usetheoai/theodb-bench
