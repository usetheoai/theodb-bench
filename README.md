# TheoDB Bench

**Reproducible performance benchmarking for TheoDB.**

Most database benchmarks answer "how fast did this run on my machine?". This one
is built to answer a stricter question:

> Can another engineer reproduce this result, verify the comparison was fair,
> inspect the raw measurements, and see exactly what was tested?

A result produced here stays valid when TheoDB **loses**. That is the point: the
framework exists to make evidence trustworthy, not to make numbers favourable.

> **Status: experimental.** The protocol is not frozen. Nothing produced before a
> release protocol is declared may be treated as an authoritative product claim.
> See [`docs/STATUS.md`](docs/STATUS.md) for exactly what works today.

---

## Quick start

Sixty seconds, no database required. The built-in `fake` system exercises the
whole pipeline in-process.

```bash
git clone <repo> && cd theodb-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

theodb-bench run vector/synthetic/smoke --system fake
```

```
run       20260813T100948Z-vector-synthetic-smoke-fake-e02b807d
status    VALID
bundle    results/20260813T100948Z-vector-synthetic-smoke-fake-e02b807d
  none                             qps=     1,784.0  recall=1.0000  (unstable)
```

That single command ran all eleven lifecycle phases and produced an immutable
run bundle. `(unstable)` is the framework telling you the repetitions disagreed
more than the threshold allows — reported, never hidden.

---

## Step by step

### 1. Check the host

Preflight decides whether this machine may measure anything, and the answer
depends on what you intend to claim.

```bash
theodb-bench doctor --profile smoke      # exit 0: fine for local validation
theodb-bench doctor --profile release    # exit 2: says which checks block you
```

```
PASS  cpu                12 logical CPUs, 10 physical cores
WARN  cpu_governor       governor is 'powersave'; frequency scaling varies results
WARN  swap               9.7 GiB swap enabled; paging silently distorts latency
PASS  cgroup_v2          cgroup v2 available
...
Host may NOT run a 'release' benchmark. Blocking: cpu_governor, swap
```

A warning that is mandatory for a profile **blocks** that profile. A release
claim measured under frequency scaling is a methodology defect, not a footnote.
To prepare a benchmark host, see
[`docs/methodology/HARDWARE.md`](docs/methodology/HARDWARE.md).

### 2. See what is available

```bash
theodb-bench list                              # benchmarks and systems
theodb-bench describe vector/synthetic/smoke   # what a benchmark actually does
```

```
vector/synthetic/smoke
  Small seeded synthetic corpus. Fast local validation of the whole pipeline;
  never a performance claim.

  corpus_size      2000
  dimension        32
  query_count      200
  k                10
  metric           l2
  seed             20260813
  warmup_queries   20
```

A system whose driver is missing stays in the list with the reason. "Not
listed" and "not installed" lead you to different actions.

### 3. Run

```bash
theodb-bench run vector/synthetic/sweep --system fake --profile smoke
```

| Flag | Meaning |
|---|---|
| `--system` | `fake`, `postgres`, `pgvector`, `theodb` |
| `--profile` | `smoke`, `pr`, `nightly`, `release`, `research` |
| `--repetitions` | overrides the benchmark default |
| `--dataset` | measure a verified dataset instead of the synthetic corpus |
| `--perf` | collect hardware counters, where the kernel permits |
| `--output` | where bundles are written (default `results/`) |

Exit code is `0` for a `VALID` or `EXPLORATORY` run and `1` for `INVALID`.

### 4. Inspect the bundle

```bash
theodb-bench validate results/<run-id>    # re-check every artifact against its schema
theodb-bench report   results/<run-id>    # re-render without re-running the workload
theodb-bench compare  results/<a> results/<b>
```

Re-analysis never re-executes the workload. That separation is why a stored run
stays useful after the analysis code changes.

### 5. Read the report

`report/report.md` leads with the status and the profile **before any number**,
so a reader who skims has already been told what the table is worth:

```markdown
**Status:** VALID · **Profile:** smoke · **Run:** `20260813T100948Z-...`

> This result is **not publishable evidence**: the profile it ran under does not
> freeze methodology or datasets.
```

---

## Running against a real database

```bash
pip install -e ".[dev,postgres]"

export PGHOST=localhost PGUSER=postgres PGDATABASE=bench
theodb-bench run vector/synthetic/sweep --system pgvector --profile pr
theodb-bench run vector/synthetic/sweep --system theodb   --profile pr
theodb-bench compare results/<pgvector-run> results/<theodb-run>
```

The adapters enforce fairness rather than trusting it: IVFFlat `lists` derives
from the real row count, `probes` is clamped to `lists`, a missing operator
class is reported unsupported rather than emitted as invalid DDL, and index use
is verified from `EXPLAIN` — forcing a scan off proves nothing on its own.

> The PostgreSQL-family adapters implement the **vector surface only** and have
> not yet been exercised against a live server. See
> [`docs/STATUS.md`](docs/STATUS.md).

---

## Running against a real dataset

A dataset is identified by its **checksums**, never by a filename. The runner
refuses to measure bytes it has not verified.

```bash
theodb-bench dataset list
theodb-bench dataset fetch sift1m
theodb-bench dataset verify sift1m

theodb-bench run vector/synthetic/sweep --system theodb --dataset sift1m
```

```
dataset   sift1m v1: verified, 1000000 vectors x 128 dims, 10000 queries
```

Declaring a dataset **requires** supplying its vectors, and supplying vectors
requires declaring their identity. Without that rule a manifest could name
`sift1m` while the run measured generated noise, and every other artifact in the
bundle would still look correct.

`datasets/manifests/` ships empty on purpose: a manifest may not carry a
checksum that was not computed from the actual bytes. To add one, fetch the
files, compute `sha256sum`, and write the manifest —
[`datasets/manifests/README.md`](datasets/manifests/README.md).

---

## What it measures

TheoDB is built for agents, so the primary surface is the **agent workload** —
what an agent exercises on every step. A system can win on per-query throughput
and lose the step ([`docs/methodology/AGENT-WORKLOAD.md`](docs/methodology/AGENT-WORKLOAD.md)).

The component surfaces below explain why the primary surface moves. They do not
substitute for it.

| Surface | Measures | Key outputs |
|---|---|---|
| **Vector ANN** | exact KNN, HNSW, IVFFlat | recall@k × QPS × latency × memory |
| **Retrieval** | lexical, dense, hybrid RRF, rerank | nDCG@10, Recall@k, MRR × performance |
| **Analytical** | row vs columnar vs Parquet | wall time, rows/s, bytes read, per-stage timing |
| **Graph** | 1/2/3-hop, BFS, fanout sweep, build | edges/s, ns/edge, bytes/edge |
| **Operations** | vectorizer under write load | foreground write latency **and** time-to-freshness |
| **AI SQL** | mock, local and remote endpoints | database vs network vs inference time, separated |

Every approximate result carries a quality axis. Throughput alone cannot
distinguish a fast system from one returning worse answers faster.

---

## Profiles

A profile declares what a result may be used for, not how fast it runs.

| Profile | Min reps | Isolation | Preflight | Publishable |
|---|---|---|---|---|
| `smoke` | 1 | optional | optional | no |
| `pr` | 3 | required | required | no |
| `nightly` | 3 | required | required | no |
| `release` | 5 | required | required | **yes** |
| `research` | 1 | optional | optional | no |

Only `release` is publishable, and it additionally requires frozen methodology,
frozen datasets, complete telemetry and a clean source tree. A `research` run is
marked `EXPLORATORY` even when technically clean.

---

## What a run produces

```
results/<run-id>/
├── manifest.json        identity, provenance, status
├── environment.json     CPU, memory, storage, toolchain, capabilities
├── benchmark.json       the declarative definition that was executed
├── system.json          capabilities and the configuration actually in force
├── validation.json      every protocol check and its outcome
├── result.json          per-configuration, per-repetition measurements
├── raw/                 telemetry, logs, client output
├── derived/             statistics, Pareto frontier, regression
└── report/              report.md and summary.json
```

Finalization freezes the manifest and every raw measurement. Re-analysis may add
new derived artifacts; it may never rewrite what was measured.

Statuses: `VALID`, `INVALID`, `EXPLORATORY`. **Invalidation is based on protocol
criteria, never on whether the number looked good.**

---

## What this framework refuses to do

These are enforced in code and covered by tests, not stated as intentions.

- **Report an unmeasured value as zero.** Four distinct absences are recorded —
  `unsupported`, `unavailable`, `not_collected`, `invalid` — because zero cache
  misses is a finding and an unavailable counter is not.
- **Fabricate an unsupported feature.** A capability the system lacks produces an
  explicit `unsupported` result, never a substituted measurement.
- **Accept a timing that came with a wrong answer.** Graph traversals and
  analytical queries are validated against an oracle the benchmark computes
  itself, before their timings are used.
- **Extend warm-up until the number improves.** The warm-up policy is declared
  and a deviation invalidates the run.
- **Compare across an incomparable baseline.** A different hardware class or
  benchmark version yields `INCOMPARABLE`, not `PASS` or `FAIL`.
- **Fail a build on a guessed threshold.** A gate whose budget was not derived
  from a measured noise floor reports `ADVISORY` and says so.
- **Attribute model latency to the database.** Inference, network and database
  time are separated in every report.
- **Delete an unfavourable run.** No code path filters a run by a metric's value.

Full rules: [`docs/methodology/`](docs/methodology/) — protocol, fairness,
statistics, hardware, publication, and the 23 measurement-integrity invariants.

---

## Repository layout

```
src/                 the theodb_bench package (flat; see CLAUDE.md)
├── adapters/        fake, postgres, pgvector, theodb
├── analysis/        quality, statistics, significance, pareto, regression, fusion
├── bench/           vector, retrieval, analytical, graph, operations
├── runner.py        the eleven-phase orchestrator
└── ...              doctor, environment, isolation, telemetry, bundle, report

schemas/             eleven versioned JSON Schemas
datasets/manifests/  dataset identity by checksum
docs/methodology/    the normative rules
docs/decisions/      ADRs
tests/               627 tests
```

---

## Development

```bash
pip install -e ".[dev]"
ruff format src tests && ruff check . && mypy && pytest -q
```

CI runs in two classes. Shared CI checks correctness on every push and its
performance numbers are **explicitly discarded** — a GitHub-hosted runner is
shared and of unknown hardware class. Real benchmarks are dispatched to a
dedicated self-hosted runner that never triggers on a pull request
([`ci/README.md`](ci/README.md)).

---

## Relationship with TheoDB

| [`theo-db`](../theo-db) | This repository |
|---|---|
| engine, Rust extension, correctness tests | public workload definitions, system-level runner |
| component-local microbenchmarks | adapters, isolation, statistics, reports, provenance |

A change to TheoDB does not require changing the methodology. A change to the
methodology creates a new benchmark version.

---

## Contributing

Contributions are welcome, particularly benchmark suites, dataset manifests,
system adapters, telemetry collectors and reproducibility fixes.

**A contribution that makes one system look faster by weakening comparability
will not be accepted.** A change that materially alters measurement semantics
must increment the benchmark version and say why.

---

## Licence

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE), which cites the
published protocols this implementation follows.
