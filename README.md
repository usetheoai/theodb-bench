# TheoDB Bench

**Open, reproducible performance benchmarking for TheoDB.**

TheoDB Bench is the public benchmarking framework for TheoDB.

It is designed to answer a stricter question than:

> "How fast did this query run on my machine?"

The goal is:

> "Can another engineer reproduce this result, verify that the comparison was fair, inspect the raw measurements, and understand exactly what was tested?"

> **Status:** experimental. The benchmark protocol is not yet frozen and results produced before a release protocol is declared MUST NOT be treated as authoritative product claims.

---

## Why a separate benchmark project?

TheoDB contains several performance-sensitive systems:

- vector KNN;
- own-code HNSW;
- own-code IVFFlat;
- vector quantization;
- lexical + vector hybrid search through RRF;
- SQL-native AI / reranking;
- native persisted-CSR graph traversal;
- an own-code columnar Table Access Method;
- own-code Parquet I/O using DataFusion + Arrow;
- a background vectorizer.

Those systems need different datasets, quality metrics, telemetry, and comparison rules.

Keeping the benchmark framework separate from the database engine gives us:

- independently versioned methodology;
- reproducible competitor adapters;
- public result schemas;
- frozen dataset manifests;
- raw result retention;
- regression tooling;
- reviewable fairness rules.

Component-local microbenchmarks still belong in the TheoDB engine repository.

---

## Principles

TheoDB Bench follows a few non-negotiable rules.

### 1. Publish evidence, not only wins

A valid benchmark result remains valid if TheoDB loses.

### 2. Compare equivalent resources

CPU, memory, storage, durability, concurrency, dataset, warm-up, and measurement policy must be documented.

### 3. Approximate search requires a quality axis

ANN throughput without recall is incomplete.

Hybrid retrieval throughput without nDCG/Recall/MRR is incomplete.

### 4. Raw measurements are part of the result

A Markdown table alone is not a benchmark artifact.

### 5. Results need provenance

Every publishable result must identify:

- TheoDB/system commit;
- benchmark commit;
- dataset checksum;
- effective configuration;
- hardware/environment;
- warm-up;
- measurement duration;
- concurrency;
- repetitions.

### 6. Microbenchmarks do not become product claims

A faster function is evidence about that function, not automatically about the database.

---

## Benchmark surfaces

TheoDB is built for agents, so the primary surface is the agent workload itself — what an agent exercises on every step.

| Primary surface | Examples | Primary outputs |
|---|---|---|
| Agent workload | step assembly, filtered memory retrieval, read-your-writes, concurrent agents | step tail latency × context quality × staleness |

Measuring only the parts does not describe the whole: a system can win on per-query throughput and lose the agent step. See `docs/methodology/AGENT-WORKLOAD.md`.

The seven component surfaces below explain why the primary surface moves. They do not substitute for it.

| Component surface | Examples | Primary outputs |
|---|---|---|
| Vector ANN | HNSW, IVFFlat, quantization | recall × QPS/latency/memory |
| Retrieval | lexical, vector, RRF, rerank | nDCG/Recall/MRR × performance |
| AI SQL | batch generation, rerank, NL-to-SQL | DB overhead + model quality |
| Graph | CSR traversal, GraphRAG | edges/s, ns/edge, p99 |
| Analytical | heap vs columnar, HTAP | query latency, CPU, I/O, GB/s |
| Lakehouse | Parquet read/aggregate | scan rate, pruning, CPU, I/O |
| Operations | ingest, vectorizer | write overhead, throughput, freshness |

---

## Benchmark levels

```text
B4  Competitive
    TheoDB vs external systems

B3  Workload
    ANN / retrieval / analytical / graph / operations

B2  Subsystem
    HNSW / IVF / columnar / Parquet / CSR / vectorizer

B1  Kernel
    distance / score / decode / merge / aggregation

B0  Environment
    CPU / RAM / NUMA / SSD / kernel / toolchain
```

B1 benchmarks are normally kept close to the engine source.

TheoDB Bench focuses primarily on B0 and B2–B4.

---

## Planned suites

### Vector

Planned workloads include:

- exact KNN baseline;
- HNSW;
- IVFFlat;
- quantized ANN;
- index build;
- concurrent search;
- ingestion;
- memory sweeps.

Metrics include:

```text
recall@1
recall@10
recall@100

QPS

p50
p95
p99
p99.9

CPU/query
cycles/query
cache misses/query

RSS
index bytes/vector
index build time
```

Headline ANN comparisons should use a matched-recall point or publish the complete Pareto frontier.

---

### Retrieval

The same corpus/query set can be executed as:

```text
lexical
vector
hybrid RRF
hybrid RRF + rerank
```

Quality:

```text
nDCG@10
Recall@k
MRR
```

Performance:

```text
QPS
p50
p95
p99
CPU
memory
stage timing
```

---

### Analytical

Planned execution paths:

```text
PostgreSQL heap
TheoDB columnar
TheoDB Parquet
```

Workloads may include TPC-H-derived and ClickBench-style suites plus TheoDB-specific analytical workloads.

TheoDB Bench will not describe a result as an official TPC result unless all applicable official requirements are actually satisfied.

---

### Graph

Planned tests:

```text
1-hop
2-hop
3-hop
BFS-like expansion
fanout sweep
graph build/rebuild
GraphRAG neighborhood expansion
```

Metrics:

```text
edges/s
ns/edge visited
p50/p95/p99
memory/edge
build time
```

---

### Vectorizer / operations

Planned tests:

```text
INSERT without vectorizer
INSERT with vectorizer
source UPDATE
backlog recovery
worker saturation
```

We separately measure:

- foreground write latency;
- embedding worker throughput;
- queue depth;
- time-to-freshness.

---

## Getting started

Requires Python 3.10+ on Linux.

```bash
git clone <repo> && cd theodb-bench
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ",postgres" for the database adapters
```

Check whether this host may measure anything:

```bash
theodb-bench doctor --profile smoke     # exit 0 means yes
theodb-bench doctor --profile release   # exit 2 means no, and says which checks block
```

Run the pipeline end to end against the built-in fake system, which needs no
database:

```bash
theodb-bench list
theodb-bench describe vector/synthetic/sweep
theodb-bench run vector/synthetic/sweep --system fake
```

That produces an immutable run bundle under `results/<run-id>/` with the
manifest, environment, validation, raw measurements, derived statistics and
both halves of the report.

```bash
theodb-bench validate results/<run-id>   # re-check every artifact against its schema
theodb-bench report   results/<run-id>   # re-render without re-running the workload
theodb-bench compare  results/<a> results/<b>
```

Against a real database:

```bash
theodb-bench run vector/synthetic/sweep --system pgvector --profile pr
theodb-bench run vector/synthetic/sweep --system theodb  --profile pr
```

Datasets are identified by checksum, never by filename:

```bash
theodb-bench dataset list
theodb-bench dataset fetch <id>
theodb-bench dataset verify <id>
```

See `docs/methodology/` for the protocol, fairness rules, statistics policy,
hardware requirements and publication preconditions, and `docs/STATUS.md` for
what is implemented today versus specified.

---

## Profiles

### `smoke`

Fast local validation.

Not suitable for public performance claims.

### `pr`

Regression detection on controlled benchmark hardware.

### `nightly`

Larger datasets, more repetitions, broader telemetry.

### `release`

Frozen methodology and publishable result bundles.

### `research`

Exploratory work. Results are explicitly non-authoritative.

---

## Result bundle

A completed run is stored as an immutable bundle.

```text
results/
└── <run-id>/
    ├── manifest.json
    ├── environment.json
    ├── benchmark.json
    ├── system.json
    ├── dataset.json
    ├── validation.json
    │
    ├── raw/
    │   ├── latency.hdr
    │   ├── client.jsonl
    │   ├── perf-stat.csv
    │   └── system.log
    │
    ├── derived/
    │   ├── statistics.json
    │   ├── quality.json
    │   └── pareto.json
    │
    └── report/
        └── report.md
```

A result can be:

```text
VALID
INVALID
EXPLORATORY
```

Invalidation is based on protocol failure, never on whether the number looks good or bad.

---

## Reproducibility

A release-grade run records at minimum:

```text
benchmark id/version
benchmark git commit

system name/version
system git commit

dataset id/version/checksum

CPU
RAM
NUMA
storage
kernel
filesystem
toolchain

database config
index config
resource limits

warm-up
duration
concurrency
repetitions

raw measurements
derived metrics
validation result
```

Dirty source trees, broken isolation, incomplete telemetry, or mismatched datasets may invalidate release runs.

---

## Fairness

Competitive reports must make relevant configuration visible.

For PostgreSQL-derived systems this can include:

```text
shared_buffers
work_mem
maintenance_work_mem
parallelism
WAL/durability settings
extension configuration
index reloptions
```

If two systems need materially different configurations, the benchmark must document why.

TheoDB-specific tuning is allowed only when equivalent system-specific tuning is also allowed for competitors under the same published policy.

---

## Statistics

TheoDB Bench reports distributions.

We do not reduce a run to one average.

Depending on the suite, reports can include:

```text
median
p50
p95
p99
p99.9
dispersion
confidence interval
throughput
quality metrics
memory
CPU
I/O
```

Hard regression gates will only be frozen after the natural noise floor of the benchmark hardware has been measured.

---

## Repository layout

Target layout:

```text
.
├── README.md
├── PRD.md
├── TRD.md
├── LICENSE
├── CHANGELOG.md
│
├── docs/
│   └── methodology/
│
├── runner/
├── adapters/
├── bench/
│   ├── vector/
│   ├── retrieval/
│   ├── analytical/
│   ├── lakehouse/
│   ├── graph/
│   ├── ai/
│   └── operations/
│
├── datasets/
├── telemetry/
├── analysis/
├── report/
├── schemas/
├── ci/
└── tests/
```

---

## Relationship with TheoDB

TheoDB Bench is not the database implementation.

The repositories have different responsibilities.

### TheoDB

Owns:

- database engine;
- extension code;
- correctness tests;
- component-local microbenchmarks;
- profiling of internal functions.

### TheoDB Bench

Owns:

- public workload definitions;
- system-level benchmark runner;
- datasets/manifests;
- external system adapters;
- isolation policy;
- statistical analysis;
- reports;
- public result provenance;
- regression policy.

A change to TheoDB does not require changing benchmark methodology.

A change to benchmark methodology creates a new benchmark version.

---

## Roadmap

### v0.1

- runner core;
- environment capture;
- dataset checksums;
- TheoDB adapter;
- PostgreSQL/pgvector baseline adapter;
- vector ANN suite;
- raw result bundle;
- Markdown/JSON report.

### v0.2

- retrieval / hybrid suite;
- quality metrics;
- Pareto and comparison reports.

### v0.3

- analytical row/columnar/Parquet suite.

### v0.4

- graph suite.

### v0.5

- vectorizer / operations suite.

### v1.0

Requires:

- frozen methodology;
- stable schemas;
- documented fairness rules;
- controlled release hardware policy;
- third-party reproduction;
- public raw artifacts for headline claims.

---

## Contributing

Contributions are welcome, especially:

- benchmark suites;
- datasets/manifests;
- system adapters;
- telemetry collectors;
- statistical validation;
- reproducibility fixes.

A contribution that makes one system look faster by weakening comparability will not be accepted.

Benchmark changes that materially alter measurement semantics must increment the benchmark version and document the reason.

---

## Result philosophy

TheoDB Bench is successful when someone can use it to prove that TheoDB is faster **or** slower under a clearly defined workload.

The framework exists to make the evidence trustworthy.

---

## License

Apache-2.0 is the intended license for TheoDB Bench.

See `LICENSE` once the repository is initialized.
