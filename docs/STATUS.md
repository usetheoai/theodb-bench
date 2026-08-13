# Implementation status

What exists, what is specified but not built, and what is blocked. Kept honest
on purpose: a README that promises a feature which does not exist is a defect,
not marketing.

Last updated: 2026-08-13.

## Working end to end

Verified by running it, not by reading the code.

| Area | State |
|---|---|
| Versioned schemas (11) with validate-before-write | **done** |
| `doctor` — 15 host checks, profile-aware blocking | **done** |
| Environment capture from procfs/sysfs, absences with reasons | **done** |
| Immutable run bundle, freeze on finalize, re-analysis allowed | **done** |
| Resource isolation + escape detection | **done** |
| Telemetry: process and perf collectors, switchable, self-measuring | **done** |
| Dataset layer: manifest, checksum, atomic fetch, verify | **done** |
| Adapter contract + fake system with 9 failure modes | **done** |
| Vector ANN workload, 11-phase orchestrator | **done** |
| Analysis: recall, nDCG, MRR, percentiles, best-of-N, aggregation, stability | **done** |
| Pareto frontier + matched quality | **done** |
| Regression comparison, fails closed, advisory thresholds | **done** |
| Reciprocal rank fusion (offline twin) | **done** |
| Retrieval suite: lexical, dense, hybrid RRF, hybrid+rerank | **done** |
| AI endpoints: mock / local / remote, with gate eligibility enforced | **done** |
| Operations suite: foreground clock vs freshness clock, 5 workloads | **done** |
| Graph suite: 1/2/3-hop, BFS, fanout sweep, build, rebuild, bounded expansion | **done** |
| Analytical suite: row vs columnar vs Parquet on identical data | **done** |
| Paired significance testing (randomisation, bootstrap CI, t-test) | **done** |
| ANN dataset readers: HDF5, fvecs, ivecs; `run --dataset` | **done** |
| Reports: markdown + machine summary | **done** |
| CLI: doctor, env, profiles, schema, list, describe, dataset, run, report, compare, validate | **done** |
| CI: shared correctness + dedicated benchmark workflow | **done** |

Run `theodb-bench run vector/synthetic/sweep --system fake` to exercise all of
it without a database.

## Written but not exercised against a real database

The PostgreSQL, pgvector and TheoDB adapters implement the **vector surface
only**, and their SQL construction, index sizing, opclass resolution and
identifier handling are covered by tests. **They have not been run against a
live server in this environment**, because none was available.

TheoDB the database has hybrid search, a columnar TAM, Parquet I/O, graph
traversal and a background vectorizer. This adapter reaches none of them: the
lifecycle methods those surfaces need are not implemented for the PostgreSQL
family. Its declared capabilities say so, and a structural test keeps them
honest.

What that means concretely:

- the fairness invariants they enforce (IVFFlat lists from the real row count,
  probes clamped to lists, opclass refusal, EXPLAIN verification) are unit
  tested but not integration tested;
- no number produced through them should be trusted until an integration run
  has happened.

To close this gap: start PostgreSQL with pgvector, then
`theodb-bench run vector/synthetic/sweep --system pgvector --profile pr`.

## Specified, not implemented

Each of these is described in the PRD/TRD and deliberately absent from the code
rather than half-present.

| Item | Where specified | Why not yet |
|---|---|---|
| Agent workload suite | `AGENT-WORKLOAD.md`, TRD §12 | The methodology is settled and its two hardest pieces now exist as parts: the freshness clock (operations suite) is read-your-writes, and the mock endpoint provides the declared non-zero model latency the surface requires. What is missing is the composite step itself, a memory schema and reference trajectories -- open questions in that document. |
| Real dataset manifests | `datasets/manifests/` | Deliberately empty. A manifest may not be committed with a checksum that was not computed from the actual bytes. The machinery is complete and proven end to end against a locally generated HDF5 file; what is missing is a download of each public corpus to compute its digest. |

## Measured against the fake, not against TheoDB

The graph, analytical, operations and retrieval suites all run end to end and
are covered by tests, but every number they have produced so far came from the
fake adapter. The fake's execution-path costs are **declared constants chosen
to give the paths their characteristic shape**, not measurements of anything.

What that buys: the suites are proven to validate answers, separate stages,
discard timings that accompany wrong results, and refuse unsupported paths.
What it does not buy: any statement about TheoDB, pgvector or PostgreSQL. The
PostgreSQL-family adapters implement the vector surface only; the graph,
analytical and operations surfaces need adapter methods those systems do not
have yet.

## Known limitations of what does exist

- **p99.9 is withheld below 1000 samples.** Correct, but it means small
  workloads report it as unavailable.
- **Memory limits are not enforced.** `apply_isolation` reports them as
  unenforceable rather than applying a cgroup, because delegated cgroup
  creation needs privileges the runner does not assume. This makes a `release`
  run INVALID on a host where the limit was declared — which is the honest
  outcome, not a workaround.
- **NUMA placement is not applied**, only detected and reported.
- **Escape detection samples procfs.** A process that forks and exits between
  two samples can be missed. This is a property of the kernel interface and is
  stated rather than hidden.
- **The synthetic corpora are synthetic.** The vector corpus is Gaussian
  noise; the retrieval corpus draws from a 28-term vocabulary with
  constructed judgements. Both exercise the pipeline and the metrics. Neither
  resembles real data, and no quality claim about any system should be read
  from them. The retrieval numbers they produce are differentiated across
  pipelines, which is what makes them useful for testing the framework and
  useless as evidence.
- **The lexical leg in the fake adapter is term-frequency scoring, not
  BM25.** It is named accordingly, so nobody compares its numbers with a real
  BM25 implementation.
- **The fake's background vectorizer is modelled, not threaded.** It consumes
  the queue at a declared rate as wall time passes, which produces the right
  shape for testing the two clocks and is not a concurrency model.

## What must never be added

For the avoidance of doubt, since these would each be an easy "improvement":

- any code path that removes or filters a run based on a metric's value;
- any default that serialises an unmeasured value as `0`;
- any comparison that proceeds when the baseline is not comparable;
- any regression gate that fails a build on a threshold nobody derived from
  measured variance.
