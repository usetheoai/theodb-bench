# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
- Regression comparison that fails closed on an incomparable baseline and
  reports ADVISORY for any threshold not derived from a measured noise floor.
- Reports in both halves: a human report that leads with status and profile,
  and a machine summary carrying provenance and limitations.
- CI in two classes: shared correctness CI whose numbers are explicitly
  discarded, and a dedicated benchmark workflow that never triggers on a pull
  request.
- Methodology documents covering the measurement-integrity invariants and the
  agent workload surface.

### Changed

- Agent workload is now the primary benchmark surface; the seven capability
  surfaces are components that explain an agent result rather than substitutes
  for it.
- Dataset manifests are JSON rather than YAML
  (`docs/decisions/0002-json-dataset-manifests.md`).

[Unreleased]: https://github.com/usetheoai/theodb-bench
