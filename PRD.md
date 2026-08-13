# TheoDB Bench — Product Requirements Document (PRD)

**Status:** Draft  
**Repository:** `theodb-bench`  
**Audience:** TheoDB maintainers, performance engineers, contributors, reviewers, users reproducing published results  
**License target:** Apache-2.0  
**Document owner:** TheoDB project  
**Last updated:** 2026-08-12

---

## 1. Executive summary

TheoDB Bench is the public, independent benchmarking project for TheoDB.

Its purpose is not to generate favorable benchmark numbers. Its purpose is to make performance claims about TheoDB **reproducible, comparable, reviewable, and falsifiable**.

TheoDB is a PostgreSQL 18-based database distribution with its own Rust extension and multiple execution/storage/search capabilities, including:

- vector similarity search;
- own-code HNSW and IVFFlat access methods;
- vector quantization;
- lexical + vector hybrid search via RRF;
- SQL-native AI functions and reranking;
- native graph traversal / GraphRAG;
- an own-code columnar Table Access Method;
- own-code Parquet I/O via DataFusion + Arrow;
- declarative vectorization with background workers.

The benchmark project MUST evaluate these capabilities with methodologies appropriate to each subsystem. A single synthetic workload or a single aggregate score is not sufficient.

The benchmark repository is intentionally separate from the TheoDB engine repository.

The split is:

- **TheoDB repository:** engine code, correctness tests, component-local microbenchmarks, developer profiling hooks.
- **TheoDB Bench repository:** system benchmarks, public datasets, workloads, competitor adapters, environment isolation, result schemas, statistical analysis, reports, and regression policies.

No historical benchmark result is automatically valid in TheoDB Bench. Results are accepted only when produced by a supported protocol and accompanied by the required provenance.

---

## 2. Problem statement

Database performance claims are easy to produce and difficult to trust.

Common benchmark failures include:

- non-equivalent hardware allocation;
- hidden cache differences;
- inconsistent warm-up;
- different durability settings;
- different dataset representations;
- selective reporting of best runs;
- averages that hide tail latency;
- quality changes hidden behind higher throughput;
- missing compiler, kernel, database, or index configuration;
- unstable datasets;
- hand-tuned competitor configurations with undocumented assumptions;
- microbenchmark gains presented as product-level gains;
- results that cannot be reproduced from source.

TheoDB spans several distinct workload families. Vector ANN, lexical retrieval, hybrid retrieval, analytical scans, Parquet execution, graph traversal, AI SQL, and background vectorization have different quality and performance metrics.

TheoDB therefore requires a benchmark system rather than a collection of ad-hoc scripts.

---

## 3. Product vision

TheoDB Bench SHOULD become the canonical evidence layer for TheoDB performance.

A valid public performance claim SHOULD be traceable through:

```text
claim
  -> report
  -> result bundle
  -> run manifest
  -> benchmark protocol
  -> source commit
  -> dataset checksum
  -> system configuration
  -> raw measurements
```

The long-term goal is:

> A technically competent third party can reproduce a TheoDB benchmark, inspect how the comparison was configured, identify uncertainty or limitations, and obtain a result within the protocol's declared variance envelope.

---

## 4. Product principles

### P1. Evidence over marketing

TheoDB Bench MUST publish unfavorable results when they are produced by a valid protocol.

The framework MUST NOT suppress cases where:

- a competitor is faster;
- a feature trades throughput for quality;
- an experimental implementation regresses;
- a feature reduces memory but increases latency;
- a benchmark exposes an architectural limit.

### P2. Equal rules

Any rule applied to TheoDB MUST apply to comparison systems unless the benchmark explicitly documents why a capability-specific exception is required.

Examples:

- same CPU allocation;
- same memory limit;
- same durability expectation;
- same dataset;
- same concurrency;
- same measurement window;
- equivalent index quality target.

### P3. Quality and performance are separate axes

For approximate retrieval, throughput MUST NOT be compared without retrieval quality.

Examples:

- ANN: recall@k × throughput / latency / memory;
- retrieval: nDCG / Recall / MRR × throughput / latency;
- NL-to-SQL: execution correctness × latency;
- vectorizer: freshness × write overhead × throughput.

### P4. Raw data is first-class

Aggregated Markdown tables are not sufficient.

A benchmark run MUST preserve machine-readable raw and derived results.

### P5. Reproducibility before breadth

A smaller number of high-quality benchmark suites is preferable to many weak suites.

### P6. Component results do not imply product results

A kernel microbenchmark MAY justify a claim about that kernel.

It MUST NOT alone justify a claim about TheoDB end-to-end performance.

### P7. Negative results remain evidence

Results SHOULD remain available even when a technique is not the default or loses to another implementation.

---

## 5. Goals

### G1. Reproducible public benchmarks

Provide a public runner capable of executing supported workloads with:

- deterministic configuration;
- environment capture;
- resource isolation;
- dataset verification;
- warm-up;
- measurement;
- telemetry;
- report generation.

### G2. Benchmark every major TheoDB performance surface

The project SHOULD cover seven major surfaces:

1. Vector ANN
2. Retrieval / hybrid search
3. AI SQL
4. Graph / GraphRAG
5. Columnar / HTAP
6. Lakehouse / Parquet
7. Operations / vectorizer / ingestion

### G3. Support competitive baselines

The architecture MUST support adapters for external systems.

Initial high-value baselines SHOULD include, where technically applicable:

- upstream PostgreSQL;
- pgvector;
- TheoDB alternative internal access paths.

Additional systems MAY be added later.

### G4. Create performance regression gates

The same framework SHOULD support:

- pull-request smoke benchmarks;
- nightly benchmarks;
- release benchmarks;
- manual research runs.

### G5. Produce immutable result bundles

Every accepted run MUST identify:

- source commit;
- benchmark commit;
- dataset identity and checksum;
- benchmark profile;
- database/system versions;
- machine information;
- resource limits;
- relevant configuration;
- warm-up policy;
- duration / operation count;
- concurrency;
- repetitions;
- raw metrics;
- derived metrics.

### G6. Make benchmark methodology reviewable

Protocol and comparison policy MUST live in version-controlled documentation.

---

## 6. Non-goals

TheoDB Bench is NOT:

- a correctness replacement for the TheoDB test suite;
- a production observability stack;
- a load-testing SaaS;
- a generic database leaderboard at launch;
- proof of production readiness;
- a guarantee that one benchmark predicts every customer workload;
- a place for undocumented one-off performance numbers;
- a mechanism for hiding losing results;
- a substitute for profiling.

HA, replication, orchestration, and control-plane benchmarking are out of initial scope unless these become explicit TheoDB product surfaces.

---

## 7. Source capability model

The benchmark design is based on the following current TheoDB capability families.

### 7.1 Vector ANN

Capabilities include:

- KNN similarity search;
- own-code HNSW;
- own-code IVFFlat;
- ScaNN-inspired asymmetric hashing carried by IVFFlat;
- multiple vector quantization kernels;
- experimental SymphonyQG.

Benchmark implication:

The suite MUST measure a quality/performance/memory frontier rather than a single QPS number.

### 7.2 Retrieval

Capabilities include:

- vector retrieval;
- lexical retrieval;
- weighted RRF hybrid fusion;
- scalar AI ranking;
- batch cross-encoder reranking;
- an own-code BM25 implementation that is not the default.

Benchmark implication:

The suite MUST be able to compare lexical, vector, hybrid, and reranked pipelines on the same corpus and query set.

### 7.3 AI SQL

Capabilities include:

- text generation from SQL;
- batched generation;
- predicates;
- sentiment analysis;
- summarization;
- ranking / reranking;
- natural-language-to-SQL and natural-language querying.

Benchmark implication:

Model quality MUST be separated from database overhead whenever the external model endpoint materially determines the result.

### 7.4 Graph

Capabilities include:

- native persisted CSR;
- graph traversal;
- GraphRAG-oriented operations.

Benchmark implication:

The suite MUST distinguish traversal kernel behavior from end-to-end GraphRAG behavior.

### 7.5 Columnar / HTAP

Capabilities include:

- own-code append-only Table Access Method;
- zone maps;
- opt-in vectorized pushdown.

Benchmark implication:

The suite MUST measure row vs columnar paths under analytical and mixed workloads.

### 7.6 Lakehouse / Parquet

Capabilities include:

- own-code Parquet read/write/aggregation;
- DataFusion + Arrow execution;
- external Parquet operation without DuckDB.

Benchmark implication:

The suite MUST expose storage/decode/pruning/aggregation behavior, not only SQL wall time.

### 7.7 Operations / vectorizer

Capabilities include:

- declarative embedding columns;
- background worker maintenance;
- embedding latency moved outside the foreground write transaction.

Benchmark implication:

The suite MUST measure foreground write overhead, backlog processing, freshness, and saturation.

---

## 8. Benchmark suite taxonomy

TheoDB Bench defines five benchmark levels.

### B0 — Environment

Purpose:

Establish whether two runs are comparable.

Captures:

- CPU model;
- logical and physical cores;
- SMT state;
- NUMA topology;
- RAM;
- storage;
- kernel;
- container runtime;
- filesystem;
- compiler/toolchain;
- PostgreSQL version;
- relevant power/governor settings;
- resource limits.

### B1 — Kernel

Purpose:

Measure isolated algorithms or functions.

Examples:

- L2/cosine/dot distance;
- quantization encode/decode;
- HNSW heap operations;
- BM25 scoring;
- RRF merge;
- CSR neighbor decode;
- column decode;
- aggregation primitives.

These benchmarks normally live with engine code but MAY be orchestrated or imported into TheoDB Bench reports.

### B2 — Subsystem

Purpose:

Measure a TheoDB subsystem through a stable interface.

Examples:

- HNSW search;
- IVFFlat probe;
- columnar scan;
- Parquet scan;
- graph traversal;
- vectorizer throughput.

### B3 — Workload

Purpose:

Measure user-visible workload behavior.

Examples:

- ANN datasets;
- retrieval datasets;
- analytical workloads;
- graph workloads;
- mixed OLTP + analytics;
- ingestion + vectorization.

### B4 — Competitive

Purpose:

Compare equivalent workloads across systems.

A B4 result MUST include comparison-policy metadata and SHOULD include a rationale for configuration equivalence.

---

## 9. Initial benchmark programs

### 9.1 Vector program

Initial suite SHOULD support:

- exact KNN baseline;
- HNSW;
- IVFFlat;
- quantized variants;
- concurrency sweeps;
- memory sweeps;
- index build;
- ingestion;
- filtered vector search when supported.

Metrics:

- recall@1 / @10 / @100;
- QPS;
- p50 / p95 / p99 / p99.9;
- CPU/query;
- cycles/query;
- instructions/query;
- cache misses/query;
- RSS;
- index bytes/vector;
- build duration;
- ingest rate;
- update rate;
- WAL bytes where relevant.

Primary result:

**Pareto frontier of retrieval quality vs throughput/latency/memory.**

### 9.2 Retrieval program

Pipelines:

1. lexical;
2. dense vector;
3. hybrid RRF;
4. hybrid RRF + rerank.

Metrics:

- nDCG@10;
- Recall@k;
- MRR;
- QPS;
- p50 / p95 / p99;
- CPU;
- memory;
- stage timing.

The framework SHOULD allow public retrieval corpora such as BEIR-style workloads, provided dataset licensing and redistribution rules are respected.

### 9.3 Analytical program

Execution paths:

- PostgreSQL heap;
- TheoDB columnar;
- TheoDB Parquet where semantically equivalent.

Workload families SHOULD include:

- TPC-H-derived execution;
- ClickBench-style analytical workloads;
- TheoDB-specific micro/macro analytical workloads.

Metrics:

- wall time;
- CPU time;
- rows/s;
- GB/s;
- bytes read;
- cache behavior;
- peak memory;
- query-level p50/p95/p99 where repeated.

TheoDB Bench MUST clearly state when a workload is "TPC-H-derived" rather than an official audited TPC result.

### 9.4 Graph program

Workloads SHOULD include:

- 1-hop traversal;
- 2-hop traversal;
- 3-hop traversal;
- BFS-like expansion;
- fanout sweeps;
- graph build/rebuild;
- GraphRAG neighborhood expansion;
- LDBC-derived workloads where applicable.

Metrics:

- edges visited/s;
- ns/edge visited;
- p50/p95/p99;
- memory/edge;
- build/rebuild time.

### 9.5 Operations program

Workloads:

- inserts without vectorizer;
- inserts with vectorizer enabled;
- source updates;
- backlog drain;
- worker saturation;
- retry / recovery scenarios.

Metrics:

- foreground write latency;
- foreground throughput;
- embeddings/s;
- queue depth;
- time-to-freshness;
- CPU;
- memory;
- failure/retry counters.

### 9.6 AI SQL program

The benchmark MUST distinguish:

- TheoDB/database overhead;
- network overhead;
- model inference latency;
- model quality.

Workloads MAY cover:

- single generation;
- batch generation;
- reranking;
- NL-to-SQL;
- sentiment;
- summarization.

Where an external model is used, the run MUST record:

- model identifier;
- endpoint type;
- relevant inference settings;
- batch size;
- network placement assumptions.

---

## 10. Benchmark profiles

### 10.1 `smoke`

Purpose:

Fast developer validation.

Characteristics:

- small deterministic datasets;
- short duration;
- minimal matrix;
- no public ranking claim.

### 10.2 `pr`

Purpose:

Detect material regressions before merge.

Characteristics:

- stable dedicated runner preferred;
- bounded runtime;
- selected sentinel workloads;
- statistical comparison against an accepted baseline.

### 10.3 `nightly`

Purpose:

Broader regression discovery.

Characteristics:

- larger datasets;
- concurrency sweeps;
- repeated runs;
- extended telemetry.

### 10.4 `release`

Purpose:

Generate publishable evidence.

Characteristics:

- frozen benchmark commit;
- frozen dataset manifests;
- full environment capture;
- multiple repetitions;
- raw artifacts retained;
- approved comparison policy;
- report generation.

### 10.5 `research`

Purpose:

Explore new techniques.

Characteristics:

- may use non-frozen parameters;
- cannot automatically become a public product claim;
- results MUST be marked experimental.

---

## 11. Result acceptance requirements

A result is **publishable** only if:

1. benchmark version is identified;
2. system source version is identified;
3. dataset identity is verified;
4. environment metadata is complete;
5. warm-up policy is satisfied;
6. measurement policy is satisfied;
7. resource isolation checks pass;
8. run completes without invalid telemetry;
9. raw output is retained;
10. statistical validation passes;
11. comparison policy passes for competitive results.

A result failing any required condition MUST be marked invalid or exploratory.

---

## 12. Statistics requirements

The benchmark framework SHOULD:

- retain every repetition;
- report distribution, not only average;
- expose median;
- expose tail latency;
- calculate variability;
- detect unstable runs;
- avoid silently removing outliers;
- record any outlier policy explicitly;
- establish environment noise before creating hard regression gates.

Regression thresholds MUST NOT be selected before the benchmark runner's natural variance has been measured.

Initial gates MAY be advisory until enough stable history exists.

---

## 13. Fairness and comparability

Competitive benchmarks MUST document:

- CPU allocation;
- RAM allocation;
- storage allocation;
- durability settings;
- database configuration;
- index configuration;
- quality target;
- preprocessing;
- data loading process;
- cache state;
- warm-up;
- concurrency;
- client placement.

Approximate indexes MUST be compared at a declared matched-quality target or through a complete Pareto curve.

If a competitor requires a materially different architecture, the report MUST explain the difference rather than forcing a misleading configuration.

---

## 14. User experience

Target CLI experience:

```bash
theodb-bench doctor

theodb-bench dataset fetch sift1m

theodb-bench run vector/sift1m \
  --system theodb \
  --profile smoke

theodb-bench compare vector/sift1m \
  --systems theodb,pgvector \
  --profile release

theodb-bench report ./results/<run-id>
```

A user SHOULD be able to go from clean checkout to a smoke result with minimal manual setup.

Release-grade runs MAY require dedicated hardware and explicit host preparation.

---

## 15. Public repository structure

```text
theodb-bench/
├── README.md
├── PRD.md
├── TRD.md
├── LICENSE
├── CHANGELOG.md
│
├── docs/
│   ├── methodology/
│   │   ├── PROTOCOL.md
│   │   ├── FAIRNESS.md
│   │   ├── STATISTICS.md
│   │   ├── HARDWARE.md
│   │   └── PUBLICATION.md
│   └── contributing/
│
├── bench/
│   ├── vector/
│   ├── retrieval/
│   ├── ai/
│   ├── graph/
│   ├── analytical/
│   ├── lakehouse/
│   └── operations/
│
├── adapters/
│   ├── theodb/
│   ├── postgres/
│   └── pgvector/
│
├── datasets/
│   ├── manifests/
│   └── licenses/
│
├── runner/
├── schemas/
├── analysis/
├── ci/
└── results/
```

`results/` MAY contain curated lightweight result manifests and reports. Large raw artifacts MAY be stored externally while remaining content-addressed and publicly retrievable.

---

## 16. Success metrics

### Product success

TheoDB Bench is successful when:

- every public TheoDB performance claim links to a reproducible result;
- third parties can reproduce release benchmarks;
- performance regressions are detected before releases;
- negative findings can be published without changing the protocol;
- adding a new system does not require rewriting benchmark suites;
- adding a new workload does not require rewriting the core runner.

### Engineering success

Targets:

- one command for smoke runs;
- machine-readable results;
- deterministic dataset manifests;
- no manual copy/paste aggregation;
- automatic environment capture;
- automatic invalid-run detection;
- versioned comparison policy;
- stable schemas.

---

## 17. Milestones

### M0 — Repository foundation

Deliver:

- repository skeleton;
- license;
- PRD;
- TRD;
- README;
- contribution policy;
- result schema draft.

### M1 — Runner core

Deliver:

- environment capture;
- process execution;
- resource limits;
- telemetry;
- result bundle;
- `doctor`.

### M2 — Vector v1

Deliver:

- first public dataset;
- TheoDB adapter;
- pgvector adapter;
- quality calculation;
- latency/throughput measurement;
- release report.

### M3 — Retrieval v1

Deliver:

- lexical;
- dense;
- hybrid;
- rerank;
- quality metrics.

### M4 — Analytical v1

Deliver:

- PostgreSQL row baseline;
- TheoDB columnar;
- TheoDB Parquet;
- selected analytical workloads.

### M5 — Graph v1

Deliver:

- traversal workloads;
- graph build;
- graph report.

### M6 — Operations v1

Deliver:

- vectorizer;
- ingest;
- freshness;
- saturation.

### M7 — Regression service

Deliver:

- PR profile;
- nightly profile;
- accepted baseline mechanism;
- regression policy.

### M8 — Public benchmark release 1.0

Gate:

- methodology frozen;
- reproducibility tested by at least one independent operator;
- release run published with raw artifacts;
- known limitations documented.

---

## 18. Risks

### R1. Benchmark becomes marketing infrastructure

Mitigation:

- public protocol;
- public raw results;
- negative result retention;
- separate benchmark repository;
- equal configuration policy.

### R2. Environment noise invalidates small deltas

Mitigation:

- dedicated runners;
- environment checks;
- repeated measurements;
- noise-floor characterization.

### R3. Too many suites before runner maturity

Mitigation:

- implement Vector v1 first;
- reuse runner for later pillars;
- block suite proliferation until schemas stabilize.

### R4. External datasets have licensing restrictions

Mitigation:

- manifests instead of redistributed data where required;
- license metadata;
- checksum validation;
- documented acquisition.

### R5. External model APIs make AI benchmarks irreproducible

Mitigation:

- record endpoint/model metadata;
- isolate database overhead;
- support local/frozen endpoints for release-grade AI tests when possible.

### R6. Competitor tuning disputes

Mitigation:

- adapter-owned configuration;
- documented tuning rationale;
- community review;
- publish configuration files verbatim.

---

## 19. Open questions

Before 1.0 the project must decide:

1. implementation language for the runner;
2. artifact storage strategy;
3. dedicated CI hardware strategy;
4. official baseline retention window;
5. benchmark result signing / attestation;
6. policy for external managed databases;
7. policy for cloud cost normalization;
8. exact dataset matrix for each release suite;
9. governance for third-party adapter changes.

---

## 20. Launch recommendation

TheoDB Bench SHOULD launch publicly as an **experimental benchmark framework**, not as an authoritative leaderboard.

The first public objective SHOULD be:

> Reproduce one rigorous vector comparison end-to-end, with complete provenance and no hidden steps.

Only after the runner, schemas, statistics, and fairness policies are stable should the project expand into a broader public database benchmark platform.
