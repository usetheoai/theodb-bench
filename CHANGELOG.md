# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Fixed

- A run manifest could name a dataset the run never measured: `dataset_id` was
  recorded while the workload generated a synthetic corpus. Declaring a dataset
  now requires supplying the vectors, and supplying vectors requires declaring
  their identity.
- The TheoDB adapter declared hybrid, lexical, columnar, Parquet, graph and
  vectorizer capabilities it does not implement, putting false claims into
  every `system.json`. It now declares only the vector surface it can exercise.

### Changed

- Agent workload is now the primary benchmark surface; the seven capability
  surfaces are components that explain an agent result rather than substitutes
  for it.
- Dataset manifests are JSON rather than YAML
  (`docs/decisions/0002-json-dataset-manifests.md`).

[Unreleased]: https://github.com/usetheoai/theodb-bench
