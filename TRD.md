# TheoDB Bench — Technical Requirements & Design (TRD)

**Status:** Draft  
**Repository:** `theodb-bench`  
**Audience:** Performance engineers, database engineers, contributors, CI maintainers  
**Last updated:** 2026-08-12

---

## 1. Purpose

This document defines the technical architecture for TheoDB Bench.

The system MUST support reproducible benchmarks across heterogeneous TheoDB capabilities while enforcing:

- deterministic configuration;
- resource isolation;
- provenance capture;
- dataset integrity;
- raw metric retention;
- comparable system adapters;
- statistical analysis;
- invalid-run detection;
- machine-readable output.

The benchmark runner is a measurement system. It MUST minimize the probability that orchestration behavior changes the result being measured.

---

## 2. Architecture overview

```text
                         +----------------------+
                         |      CLI / API       |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Run Orchestrator   |
                         +----------+-----------+
                                    |
             +----------------------+----------------------+
             |                      |                      |
             v                      v                      v
     +---------------+      +---------------+      +---------------+
     | Dataset Layer |      | System Adapter|      | Workload Spec |
     +-------+-------+      +-------+-------+      +-------+-------+
             |                      |                      |
             +----------------------+----------------------+
                                    |
                                    v
                         +----------------------+
                         | Isolation / Launcher |
                         +----------+-----------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
        +------------------+                  +------------------+
        |  System Under    |                  | Load Generator / |
        |      Test        |                  | Benchmark Client |
        +------------------+                  +------------------+
                 |                                     |
                 +------------------+------------------+
                                    |
                                    v
                         +----------------------+
                         | Telemetry Collectors |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Raw Run Bundle    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Stats / Validation   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Report / Comparison  |
                         +----------------------+
```

---

## 3. Core design decisions

### D1. Benchmark definitions are declarative

Each benchmark SHOULD have a machine-readable specification.

Example:

```yaml
id: vector/sift1m/hnsw
version: 1

dataset:
  id: sift1m
  manifest: datasets/manifests/sift1m.yaml

workload:
  type: ann
  k: [1, 10, 100]
  concurrency: [1, 8, 32]
  duration_seconds: 300
  warmup_seconds: 60

quality:
  metric: recall
  ground_truth: dataset

telemetry:
  latency: hdr
  perf_stat: true
  process: true
  postgres: true

repetitions: 5
```

The exact schema MAY evolve, but benchmark behavior MUST NOT depend on undocumented shell state.

### D2. Systems use adapters

A benchmark suite MUST NOT contain system-specific lifecycle logic.

System adapters own:

- installation/bootstrap;
- start/stop;
- schema setup;
- data loading;
- index creation;
- query execution primitives;
- configuration export;
- system-specific stats capture;
- teardown.

Example interface:

```text
prepare()
start()
wait_ready()
load_dataset()
build_index()
run_query()
collect_stats()
export_config()
stop()
cleanup()
```

### D3. Run bundles are immutable

A successful execution creates a unique run directory.

No result file is modified after finalization.

If post-processing changes, new derived artifacts MAY be created but original raw measurements MUST remain intact.

### D4. Result schemas are versioned

Every machine-readable artifact MUST contain a schema version.

### D5. Dataset identity is content-based

A dataset MUST be identified by:

- logical name;
- version;
- source;
- license metadata;
- checksum(s);
- preprocessing version.

### D6. Quality is calculated independently of the system

Where possible, recall/nDCG/MRR computation SHOULD be performed by benchmark-owned analysis code rather than trusted from the system under test.

### D7. Orchestration and reporting are separated

A run MAY be analyzed again without rerunning the workload.

---

## 4. Repository layout

```text
theodb-bench/
├── README.md
├── PRD.md
├── TRD.md
│
├── cmd/
│   └── theodb-bench/
│
├── runner/
│   ├── orchestrator/
│   ├── launcher/
│   ├── isolation/
│   ├── lifecycle/
│   └── validation/
│
├── adapters/
│   ├── theodb/
│   ├── postgres/
│   └── pgvector/
│
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
│   ├── manifests/
│   ├── transforms/
│   └── licenses/
│
├── telemetry/
│   ├── process/
│   ├── linux_perf/
│   ├── postgres/
│   ├── io/
│   └── latency/
│
├── analysis/
│   ├── stats/
│   ├── ann/
│   ├── retrieval/
│   ├── regression/
│   └── pareto/
│
├── report/
│   ├── markdown/
│   ├── json/
│   └── charts/
│
├── schemas/
│   ├── benchmark.schema.json
│   ├── environment.schema.json
│   ├── manifest.schema.json
│   └── result.schema.json
│
├── ci/
│   ├── smoke/
│   ├── pr/
│   ├── nightly/
│   └── release/
│
└── tests/
```

---

## 5. CLI

Proposed command surface:

```text
theodb-bench doctor
theodb-bench list
theodb-bench describe <benchmark>

theodb-bench dataset fetch <dataset>
theodb-bench dataset verify <dataset>

theodb-bench system doctor <system>
theodb-bench system config <system>

theodb-bench run <benchmark>
theodb-bench compare <benchmark>

theodb-bench validate <run-dir>
theodb-bench analyze <run-dir>
theodb-bench report <run-dir>
```

Example:

```bash
theodb-bench compare vector/sift1m \
  --systems theodb,pgvector \
  --profile release \
  --output results/
```

---

## 6. Run lifecycle

A run MUST execute the following phases.

### Phase 0 — Preflight

Checks:

- supported OS/kernel;
- required tools;
- CPU topology;
- CPU governor;
- memory availability;
- storage availability;
- time synchronization;
- container/runtime support if needed;
- exclusive benchmark lock;
- dataset checksum;
- system binary/version.

Failure:

Preflight failure MUST stop the run before measurement.

### Phase 1 — Environment capture

Capture:

```text
hostname
timestamp
CPU
NUMA topology
SMT
RAM
kernel
filesystem
storage devices
mount options
compiler
container runtime
PostgreSQL
TheoDB
benchmark commit
system commit
git dirty state
```

Dirty source trees SHOULD invalidate release runs unless explicitly allowed by policy.

### Phase 2 — Isolation setup

Possible controls on Linux:

- cpuset/cgroup assignment;
- CPU quota;
- CPU affinity;
- memory limit;
- swap policy;
- NUMA binding;
- process tree tracking;
- I/O limits where applicable;
- network namespace where appropriate.

Release profiles SHOULD prefer dedicated bare-metal or dedicated VM runners.

### Phase 3 — System bootstrap

Adapter:

- starts system;
- waits for readiness;
- records effective configuration;
- validates durability assumptions;
- creates benchmark database/schema.

### Phase 4 — Dataset load

Requirements:

- dataset verified before use;
- preprocessing deterministic;
- load duration measured separately from query benchmark unless load is the benchmark;
- loaded row/vector counts validated.

### Phase 5 — Index/build

Index build MUST be separately measurable.

Capture:

- index configuration;
- build time;
- peak memory if available;
- index size;
- database size;
- WAL generated where relevant.

### Phase 6 — Warm-up

Warm-up policy is benchmark-specific.

Warm-up MUST NOT be silently extended until a desired number appears.

Possible policies:

- fixed time;
- fixed operation count;
- defined cache-state transition.

### Phase 7 — Measurement

The runner starts synchronized telemetry before measurement and stops it after the measurement window.

No benchmark result SHOULD rely on the client-reported average alone.

### Phase 8 — Cooldown / repetition

For repeated runs the suite MUST define whether:

- indexes are rebuilt;
- process is restarted;
- cache is dropped;
- database is restored;
- only the client is restarted.

This is part of benchmark semantics.

### Phase 9 — Validation

Validate:

- expected operation count;
- no SUT crash;
- no client crash;
- no timeout rate above policy;
- no invalid result rows;
- CPU limit respected;
- memory limit respected;
- no unexpected helper process outside isolation;
- quality output valid;
- telemetry complete.

### Phase 10 — Finalization

Create immutable run bundle and mark:

```text
VALID
INVALID
EXPLORATORY
```

---

## 7. Result bundle

Canonical shape:

```text
results/
└── 2026-08-12T231000Z-vector-sift1m-theodb-abc123/
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
    │   ├── process.csv
    │   ├── postgres.json
    │   └── system.log
    │
    ├── derived/
    │   ├── statistics.json
    │   ├── quality.json
    │   ├── pareto.json
    │   └── regression.json
    │
    └── report/
        ├── report.md
        └── summary.json
```

---

## 8. Run manifest

Required fields SHOULD include:

```json
{
  "schema_version": 1,
  "run_id": "2026-08-12T231000Z-vector-sift1m-theodb-abc123",
  "status": "VALID",
  "benchmark": {
    "id": "vector/sift1m/hnsw",
    "version": 1,
    "profile": "release",
    "benchmark_commit": "..."
  },
  "system": {
    "id": "theodb",
    "version": "...",
    "commit": "...",
    "dirty": false
  },
  "dataset": {
    "id": "sift1m",
    "version": "...",
    "sha256": "..."
  },
  "execution": {
    "warmup_seconds": 60,
    "measurement_seconds": 300,
    "concurrency": 32,
    "repetition": 3
  },
  "environment_ref": "environment.json",
  "system_ref": "system.json",
  "validation_ref": "validation.json"
}
```

---

## 9. Environment schema

Minimum CPU metadata:

```text
vendor
model
microarchitecture if detectable
sockets
physical cores
logical CPUs
SMT
frequency policy
NUMA nodes
cache hierarchy if available
```

Memory:

```text
total
NUMA distribution
hugepages
swap
```

Storage:

```text
device
type
filesystem
mount options
capacity
```

Software:

```text
kernel
glibc
compiler
rust toolchain
postgres
theodb
benchmark runner
container runtime
```

---

## 10. Isolation requirements

### 10.1 CPU

The runner SHOULD support:

- explicit allowed CPU set;
- SUT affinity;
- client affinity;
- optional separation of SUT and client cores.

A release benchmark MUST record the allocation.

### 10.2 Process tree

The runner MUST track child processes.

A process created by the SUT or adapter MUST NOT escape declared resource controls.

### 10.3 Memory

The runner SHOULD support hard or monitored memory bounds.

OOM events invalidate the run unless OOM behavior is explicitly the benchmark target.

### 10.4 NUMA

Large memory/vector benchmarks SHOULD define NUMA placement.

Cross-NUMA effects MUST be captured or explicitly uncontrolled.

### 10.5 Client placement

For local benchmarks, the client and SUT CPU sets SHOULD be separately declared.

For networked benchmarks, network topology MUST be documented.

---

## 11. Timing and latency

Latency collection SHOULD use a high-dynamic-range histogram or equivalent raw representation.

Required percentiles:

- p50;
- p95;
- p99;
- p99.9 when sample size supports it.

The framework SHOULD preserve:

- successful latency;
- timeout count;
- error count;
- achieved throughput.

Coordinated omission MUST be considered for load-generation modes that target a fixed request rate.

Closed-loop and open-loop workloads MUST be identified explicitly.

---

## 12. Agent workload benchmark design

This is the **primary** benchmark surface. TheoDB is built for agents, and the sections that follow (§13–§18) describe the component surfaces that explain why this one moves.

Full methodology: `docs/methodology/AGENT-WORKLOAD.md`.

### 12.1 Unit of measurement

The unit is the **agent step**, not the query. A step assembles context from the database through several retrievals (vector, lexical, graph — filtered by agent/session/tenant), calls a model, and writes back observations that must be retrievable on the next step.

The database is exercised in context assembly and write-back. Model latency MUST NOT be attributed to the database (§14, §18).

### 12.2 Headline metric

**Step assembly latency:** p50/p95/p99/p99.9 of the composite of every database call within one step — not the throughput of any individual leg.

A per-leg breakdown MUST accompany it, so a regression is attributable. The composite MUST be measured under real intra-step concurrency; summing independently measured legs reports a step that never happened.

### 12.3 Required measurements

Deterministic, gate-able, executed against the `mock` endpoint (§18):

```text
step_assembly_p50/p95/p99/p99.9
per_leg_breakdown
filtered_recall × selectivity sweep
filtered_latency × selectivity sweep
read_your_writes_staleness_window
read_your_writes_failure_rate @ step gap
concurrent_agents sweep (M agents, disjoint working sets)
read_write_interference_delta (loaded vs quiesced)
```

Model-dependent, executed against a `local` frozen model, marked environment-dependent and never backing an automatic gate:

```text
task_success_rate
cost_per_step (tokens + database CPU)
```

### 12.4 Filtered retrieval is the default case

Per-tenant/session filtering applies to essentially every agent retrieval. Filter selectivity MUST be a swept axis, never a single point: graph-based indexes degrade sharply as selectivity drops, and pre-filter versus post-filter strategy dominates the result.

### 12.5 Read-your-writes is a correctness property

A benchmark spec MUST declare a staleness bound per pipeline. "Eventually retrievable" is not a bound.

An observation written at step *N* and not retrievable within the declared bound is a **lost write from the agent's perspective**, not a slow result. Violating the bound marks the run `INVALID` at phase 9 (§6), not merely unfavorable.

### 12.6 Warm-up policy

Many agents holding small working sets is a cold-ish regime by construction. Warm-up MUST NOT load every tenant's data into cache — that measures a system that will not exist in production. The warm-up policy MUST be declared and MUST be justified against the working-set model of the workload.

### 12.7 Failed steps

Timeouts, empty retrievals, and constraint violations are counted and reported separately. A failed step MUST NOT enter the latency distribution as if it had succeeded.

### 12.8 Mock endpoint profile

The `mock` endpoint MUST have a declared, non-zero latency profile. A zero-latency model changes the concurrency regime of the whole loop and flatters any system that overlaps I/O with inference.

---

## 13. Vector benchmark design

### 13.1 Correctness ground truth

Each ANN dataset MUST provide or generate exact nearest-neighbor ground truth.

### 13.2 Parameter sweeps

HNSW example dimensions:

```text
ef_search
concurrency
dataset size
k
quantization
memory budget
```

IVF example dimensions:

```text
lists
probes
concurrency
quantization
k
```

### 13.3 Required outputs

Per configuration:

```text
recall@k
qps
p50
p95
p99
p99.9
rss
index_size
cpu_seconds
cycles
instructions
cache_misses
```

### 13.4 Pareto calculation

A point is dominated when another configuration is:

- equal or better in quality;
- equal or better in target performance metric;
- strictly better in at least one.

Reports SHOULD display the non-dominated frontier.

### 13.5 Matched-recall comparison

When presenting a headline QPS comparison, the report MUST state the target recall and the interpolation/selection method.

---

## 14. Retrieval benchmark design

Dataset abstraction:

```text
documents
queries
relevance judgments
```

Supported pipelines:

```text
lexical
vector
hybrid_rrf
hybrid_rrf_rerank
```

Required quality metrics SHOULD include:

- nDCG@10;
- Recall@10 / @100;
- MRR where appropriate.

Stage timing MAY include:

```text
lexical retrieval
vector retrieval
fusion
rerank
serialization
```

The benchmark MUST NOT attribute external model latency to the database engine without separating it in the report.

---

## 15. Analytical benchmark design

Execution modes:

```text
postgres_heap
theodb_columnar
theodb_parquet
```

Workload abstraction:

```text
schema
data generator/import
queries
expected result checks
scale factor
```

Each query result MUST be correctness-validated before accepting timing.

Recommended telemetry:

- wall duration;
- CPU duration;
- rows processed;
- bytes read;
- block/page statistics;
- process RSS;
- perf counters where supported.

The project MUST avoid using official TPC branding in a way that implies audited compliance unless actual TPC rules and audit requirements are satisfied.

---

## 16. Graph benchmark design

Dataset fields:

```text
vertices
edges
edge directionality
edge types
properties
expected traversal results
```

Workloads:

```text
neighbors
k-hop
BFS
fanout sweep
graph build
rebuild
GraphRAG neighborhood expansion
```

Metrics:

```text
edges_visited
edges_per_second
ns_per_edge
latency percentiles
rss
bytes_per_edge
build time
```

---

## 17. Vectorizer benchmark design

Two clocks must be measured:

### Foreground clock

Time experienced by the write transaction.

### Freshness clock

Time until the embedding derived from that write becomes queryable/up-to-date.

Required measurements:

```text
write p50/p95/p99
write throughput
worker throughput
queue depth
time-to-freshness p50/p95/p99
retry count
failed jobs
```

Saturation tests SHOULD increase foreground write rate until worker backlog grows persistently.

---

## 18. AI SQL benchmark design

External inference creates reproducibility risk.

The runner SHOULD support three endpoint modes:

### `mock`

Deterministic endpoint for measuring database/control-plane overhead.

### `local`

Frozen local model or service for repeatable end-to-end testing.

### `remote`

Real hosted model endpoint.

Remote mode MUST be marked environment-dependent and SHOULD NOT be used for strict performance regression gates.

NL-to-SQL quality runs SHOULD validate generated/executed SQL against an expected answer set where the dataset supports it.

---

## 19. PostgreSQL telemetry

Where available, the TheoDB/PostgreSQL adapter SHOULD capture:

- `EXPLAIN (ANALYZE, ...)` for targeted diagnostic modes;
- `pg_stat_statements`;
- `pg_stat_io`;
- relation/index sizes;
- WAL deltas;
- buffer/cache statistics;
- relevant extension statistics.

Telemetry queries MUST be designed to avoid materially perturbing short benchmark windows.

For release runs, telemetry overhead SHOULD be measured or collectors MAY run only between repetitions.

---

## 20. Linux telemetry

Optional collectors:

```text
perf stat
process RSS
context switches
page faults
disk counters
CPU utilization
frequency
thermal state
```

Perf event availability differs by kernel and host policy. Missing events MUST be recorded rather than silently reported as zero.

---

## 21. Statistical processing

For each benchmark point:

1. retain all repetitions;
2. validate each repetition;
3. compute median;
4. compute dispersion;
5. compute requested percentiles;
6. detect instability;
7. compare to baseline if applicable.

The framework SHOULD support confidence intervals where statistically meaningful.

The project MUST NOT delete a run because it is unfavorable.

Invalidation MUST be based on protocol criteria, not the measured outcome.

---

## 22. Regression model

A baseline is identified by:

```text
benchmark id
benchmark version
profile
system
hardware class
accepted commit
```

Regression comparison MUST fail closed when the baseline is not comparable.

Example:

```yaml
regression:
  throughput:
    max_regression_pct: 3
  p99:
    max_regression_pct: 5
  rss:
    max_regression_pct: 5
  recall_at_10:
    max_absolute_regression: 0.001
```

These thresholds are illustrative.

Actual thresholds MUST be derived after measuring runner variance.

---

## 23. Profiles

### Smoke

```text
duration: short
repetitions: 1
dataset: small
telemetry: minimal
isolation: best-effort
publishable: no
```

### PR

```text
duration: bounded
repetitions: >= 3 where practical
dataset: sentinel
telemetry: standard
isolation: required on benchmark runner
publishable: no
regression_gate: yes
```

### Nightly

```text
duration: medium/long
repetitions: multiple
datasets: broader
telemetry: extended
regression_gate: yes
```

### Release

```text
duration: full
repetitions: protocol-defined
datasets: frozen
telemetry: full
environment: dedicated
raw_artifacts: retained
publishable: yes
```

---

## 24. CI architecture

Normal GitHub-hosted shared runners SHOULD NOT be treated as authoritative for small performance changes.

Recommended split:

```text
GitHub PR
   |
   +-- correctness / schema / unit tests
   |
   +-- optional smoke performance
   |
   +-- dispatch to dedicated benchmark runner
                  |
                  +-- PR benchmark
                  +-- nightly
                  +-- release
```

A dedicated runner SHOULD have:

- known hardware;
- controlled background services;
- stable kernel;
- stable power settings;
- exclusive benchmark lock;
- storage reset/cleanup policy.

---

## 25. Security

Benchmark adapters execute database binaries and external processes.

Requirements:

- no untrusted benchmark contribution automatically executes on privileged release hardware;
- secrets MUST NOT be written into result bundles;
- remote AI endpoint credentials MUST be redacted;
- dataset download checksums MUST be verified;
- arbitrary shell hooks SHOULD be restricted in trusted profiles;
- release runner permissions SHOULD follow least privilege.

---

## 26. Dataset management

Manifest example:

```yaml
id: example
version: "1"
license: "..."
source: "..."
files:
  - path: base.fvecs
    sha256: "..."
  - path: query.fvecs
    sha256: "..."
preprocess:
  version: 1
```

Commands:

```bash
theodb-bench dataset fetch example
theodb-bench dataset verify example
```

Large public datasets SHOULD NOT be committed directly to Git.

---

## 27. Adapter contract

Each adapter MUST expose machine-readable capabilities.

Example:

```json
{
  "system": "theodb",
  "capabilities": {
    "vector_hnsw": true,
    "vector_ivfflat": true,
    "hybrid": true,
    "columnar": true,
    "parquet": true,
    "graph": true
  }
}
```

Unsupported workload features MUST produce a clear "unsupported" result, not a failed or fabricated measurement.

---

## 28. Fair configuration contract

Adapters SHOULD export the full effective configuration.

For PostgreSQL-derived systems this SHOULD include relevant settings such as:

```text
shared_buffers
work_mem
maintenance_work_mem
max_connections
wal settings
checkpoint settings
parallelism
extension settings
index reloptions
```

The comparison report MUST show material differences between systems.

---

## 29. Testing the benchmark itself

TheoDB Bench requires its own tests.

### Unit tests

- schema validation;
- statistics;
- Pareto logic;
- result parsing;
- checksum logic;
- latency histogram logic.

### Integration tests

- fake SUT;
- process tree enforcement;
- timeout handling;
- crash handling;
- dataset fetch;
- run finalization.

### Golden tests

- deterministic report generation;
- known regression classification;
- known ANN quality calculation.

### Self-benchmark

The runner SHOULD measure its own overhead for selected collectors.

---

## 30. Publication model

A published result SHOULD consist of:

```text
human report
machine summary
run manifest
raw artifacts
configuration
dataset manifest
benchmark commit
system commit
known limitations
```

Reports SHOULD be content-addressable where practical.

A public result MUST NOT be silently overwritten.

Corrections SHOULD create a new report and mark the earlier result superseded.

---

## 31. Initial implementation sequence

### Stage 1 — Core

Implement:

- CLI;
- manifest schema;
- environment capture;
- dataset verification;
- lifecycle;
- result bundle.

### Stage 2 — Isolation

Implement:

- CPU affinity/cpuset;
- memory tracking;
- process tree validation;
- exclusive lock.

### Stage 3 — Vector

Implement:

- SUT adapter;
- pgvector adapter;
- ANN dataset parser;
- exact ground truth / recall;
- latency;
- throughput;
- Pareto.

### Stage 4 — Reporting

Implement:

- Markdown;
- JSON summary;
- comparison tables;
- regression output.

### Stage 5 — CI

Implement:

- PR profile;
- nightly;
- baseline comparison.

Only then expand to retrieval, analytical, graph, operations, and AI.

---

## 32. Definition of done for v0.1

`theodb-bench v0.1` is complete when a third party can:

1. clone the repository;
2. run `doctor`;
3. fetch/verify the first vector dataset;
4. bootstrap TheoDB and one baseline adapter;
5. execute a defined ANN benchmark;
6. receive raw latency and quality measurements;
7. receive environment metadata;
8. receive a validation status;
9. generate a comparison report;
10. reproduce the procedure from documentation without undocumented manual steps.

---

## 33. Definition of done for 1.0

1. schemas are stable;
2. benchmark protocol is versioned;
3. fairness policy is versioned;
4. release hardware policy is documented;
5. at least vector, retrieval, analytical, and graph suites are supported;
6. external reproduction has been demonstrated;
7. raw artifacts are publicly available for headline claims;
8. regression pipeline is operational;
9. negative results are preserved;
10. published claims link to exact run bundles.
