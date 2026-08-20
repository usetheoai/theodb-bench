# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It complements — does not replace — the global rules in `~/.claude/CLAUDE.md`. On conflict, the more specific rule wins.

**Everything committed to this repository is written in English:** code, identifiers, comments, docstrings, documentation, `CHANGELOG.md`, commit messages, PR bodies, issue titles, and CLI output strings.

---

## Current state (check before assuming)

**The repository has no code and no commits yet.** Only `PRD.md`, `TRD.md` and `README.md` exist — three specification documents. There is no `LICENSE`, no `CHANGELOG.md`, no `Makefile`, no build manifest, and no tests.

Practical consequences:

- **There are no build/lint/test commands.** Do not invent them, and do not run `theodb-bench ...` commands — they describe the *intended* interface (README "Proposed CLI", TRD §5), not an existing binary.
- When the first code lands, M0 (PRD §17) also requires `LICENSE` (Apache-2.0) and `CHANGELOG.md`.

Where this file diverges from the repository, the repository wins — and this file should be updated.

## Settled decisions

Recorded so they are not relitigated. Owner decisions, 2026-08-12.

| Question | Decision |
|---|---|
| Runner language (was PRD §19 open question 1) | **Python** |
| Canonical repository layout | **TRD §4**; PRD §15 and the README converge to it |
| Primary benchmark surface | **Agent workload** (TRD §12, PRD §9.0, `docs/methodology/AGENT-WORKLOAD.md`) |
| Relationship to the predecessor harness | **Rewrite**, guided by the recorded spec — not a port |
| Measurement integrity invariants | Adopted **in full** as `docs/methodology/MEASUREMENT-INTEGRITY.md` |

### This is a new project, not a continuation

A predecessor measurement harness existed inside `theo-db` and was removed on 2026-08-12 (commit `7cd157d`, 268 files, recoverable via `git restore -- benchmarks`). A reconstruction spec for it lives at `theo-db/.claude/knowledge-base/benchmarks-reconstruction-spec.md`.

**Treat that material as evidence about how measurements fail, not as a design to inherit.** Its module decomposition, CLI surface, CI wiring, dependency set, and pillar breakdown belong to a project that lived inside the engine repo and aimed at a pgvector head-to-head. Reproducing that shape here would build the second version of a finished project instead of the first version of this one.

What carries over is exactly one thing: the 23 measurement-integrity invariants, adopted in full in `docs/methodology/MEASUREMENT-INTEGRITY.md`. Each came from a real defect, and four of them are phrased in competitor-specific terms — the document says which, and says to restate rather than assume them when adding a new adapter.

---

## What this project is

TheoDB Bench is the **public** benchmarking framework for TheoDB — a database built for **agents** (the engine lives in the sibling repo `../theo-db`, a Rust extension on PostgreSQL 18).

The central thesis, and the one thing to internalize before writing any line:

> The goal is not to produce favorable numbers. It is to produce **reproducible, comparable, reviewable, falsifiable** evidence — including when TheoDB loses.

A result showing TheoDB slower, produced by a valid protocol, is a **valid** result and must be published (PRD P1/P7).

The second thing: **the unit of measurement is the agent step, not the query.** A system can win on per-query throughput and lose the step that an agent actually takes. Every widely used database benchmark measures the query; that gap is what this project exists to close.

### Split of responsibility with `theo-db`

| `theo-db` (sibling) | `theodb-bench` (here) |
|---|---|
| engine, Rust extension, correctness tests | public workload definitions, system-level runner |
| component-local microbenchmarks (B1) | external system adapters, isolation, statistics, reports, provenance |

A change to TheoDB does **not** require changing the methodology. A change to the methodology **creates a new benchmark version** (README "Relationship with TheoDB").

---

## Architecture (the pipeline)

The whole design revolves around a linear pipeline (TRD §2), and the boundaries between the boxes are what buys reproducibility:

```
CLI → Orchestrator → {Dataset Layer | System Adapter | Workload Spec}
    → Isolation/Launcher → {SUT | Load Generator}
    → Telemetry Collectors → Raw Run Bundle → Stats/Validation → Report
```

The seven design decisions (TRD §3, D1–D7) that constrain any implementation:

- **D1 — benchmarks are declarative.** Machine-readable specification (YAML). Benchmark behavior may **never** depend on undocumented shell state.
- **D2 — systems enter through an adapter.** A benchmark suite **must not** contain system-specific lifecycle logic. Every `prepare/start/wait_ready/load_dataset/build_index/run_query/collect_stats/export_config/stop/cleanup` belongs to the adapter (TRD §26).
- **D3 — run bundles are immutable.** Nothing is modified after finalization. New post-processing creates new *derived* artifacts; the original `raw/` stays intact.
- **D4 — every machine-readable artifact carries a `schema_version`.**
- **D5 — dataset identity is content-based** (name + version + source + license + checksum + preprocessing version).
- **D6 — quality is computed by the benchmark, not by the SUT.** recall/nDCG/MRR are computed by our own code; do not trust the number the system under test reports.
- **D7 — orchestration and reporting are separated.** `analyze`/`report` re-run over an existing bundle without re-executing the workload.

### Run lifecycle (TRD §6)

Eleven mandatory phases: `0 preflight → 1 environment capture → 2 isolation → 3 bootstrap → 4 dataset load → 5 index build → 6 warm-up → 7 measurement → 8 cooldown/repetition → 9 validation → 10 finalization`.

The points most often violated by carelessness, which **cannot** be:

- Preflight failure **stops the run before any measurement**.
- Index build is measurable **separately** from the query benchmark.
- Warm-up may **not** be silently extended until the desired number appears.
- Repetition semantics (rebuild the index? restart the process? drop caches? restore the database?) is part of the benchmark definition, not of the operator's mood.
- Finalization marks `VALID | INVALID | EXPLORATORY`.

### Invalidation

**Invalidation is always by protocol criteria, never by the measured outcome** (TRD §20, PRD §11). A run is not discarded for being unfavorable. Never write code, a gate, or a heuristic that removes or filters a run based on the value of a metric.

---

## Non-negotiable invariants

Two documents hold simultaneously and neither subsumes the other:

- **`docs/methodology/MEASUREMENT-INTEGRITY.md`** — 23 invariants on what makes an individual *number* trustworthy (recall semantics, forced-and-verified index use, paired significance testing, seeded determinism, provenance). Read it before writing anything that measures, compares, or reports.
- **`docs/methodology/AGENT-WORKLOAD.md` §5** — 7 further traps specific to the agent surface (never fold model time into a step, compose from real concurrency rather than a sum of legs, do not warm what production will not have warm).

Below are the project-level invariants derived from the PRD and TRD — what may be published, what invalidates a run, how systems are compared. These are the points where a reasonable-looking shortcut destroys the purpose of the project:

1. **Never fabricate or infer a measurement.** A feature an adapter does not support yields an explicit `unsupported` result — not a failure and not an invented number (TRD §26). An unavailable perf counter is **recorded as missing**, never reported as zero (TRD §19).
2. **Approximate throughput without a quality axis is incomplete.** ANN without recall@k, hybrid retrieval without nDCG/Recall/MRR — do not publish. A headline QPS comparison requires a stated target recall plus the interpolation method, or the complete Pareto frontier (TRD §12.5).
3. **Do not attribute external model latency to the database.** In AI SQL / rerank, database overhead, network, and inference are separated in the report (TRD §13, §17).
4. **Secrets never enter a result bundle.** Remote endpoint credentials are redacted (TRD §24).
5. **No official TPC branding** unless the actual requirements and audit are genuinely satisfied — use "TPC-H-derived" (README, TRD §14, PRD §9.3).
6. **An average alone is not a result.** We report the distribution: p50/p95/p99/p99.9, dispersion, every repetition retained. An outlier is removed only under an explicit, recorded policy.
7. **Regression thresholds only after measuring the runner's own variance** (TRD §21, PRD §12). The numbers in TRD §21 are illustrative. Regression comparison **fails closed** when the baseline is not comparable.
8. **A published result is never silently overwritten.** A correction creates a new report and marks the previous one superseded (TRD §29).
9. **Equal rules.** Any rule applied to TheoDB applies to competitors; TheoDB-specific tuning is allowed only if equivalent tuning is allowed for the competitor under the same published policy (README "Fairness", PRD P2).
10. **A microbenchmark does not become a product claim** (PRD P6).
11. **Every publishable number comes out of `theodb-bench run`.** A script that talks to an adapter directly
    produces a number with no bundle, no schema validation, no environment record and no immutable artefact —
    which means nobody outside the room can reproduce it. Owner's instruction, 2026-08-17. Ad-hoc queries are
    fine for *diagnosis* (reading `pg_settings`, reading a plan); the moment a figure is reported, it comes
    from a run. This also keeps the harness honest: three defects were found on 2026-08-17 only because a
    measurement was forced through it — a single `statement_timeout` that cancelled million-row index builds,
    an abort classifier that blamed the database for the harness's own refusals, and a rescore knob that could
    not be swept.
12. **Equal rules apply to configuration, and a default is not an operating point.** Invariant 9 says any rule
    applied to TheoDB applies to competitors; this is the half that gets missed, because the unfair
    configuration usually looks like *no* configuration. Measured in one session on 2026-08-17, four times:
    AlloyDB's `scann.num_leaves_to_search` does nothing without `LOAD 'alloydb_scann'`; `quantizer='AH'` fails
    unless `scann.enable_ah_quantizer` is on **at build time**, so the default builds SQ8; ScaNN's rescore
    (`pre_reordering_num_neighbors`) ships at `-1` and caps recall at 0.6568 where 100 gives 0.9964; and on our
    own side `theodb.enable_columnar_agg` ships off, which is a 13x difference on the same table and the same
    query. Every one of those produced a `VALID` bundle with a plausible frontier. Before a cross-engine
    number is reported, each engine's quality-critical settings are **verified in force**, and the report
    states them. Publishing a competitor measured at a crippled default is the most dangerous error available
    here, because it flatters us.

---

## Implementation order

TRD §30 defines the sequence, and PRD R3 ("too many suites before runner maturity") is the explicit risk it mitigates. **Do not open new suites before the core stabilizes:**

```
Stage 1 Core       CLI, manifest schema, environment capture, dataset verification, lifecycle, result bundle
Stage 2 Isolation  CPU affinity/cpuset, memory tracking, process-tree validation, exclusive lock
Stage 3 Vector     TheoDB adapter, pgvector adapter, ANN dataset parser, exact ground truth/recall, latency, throughput, Pareto
Stage 4 Reporting  Markdown, JSON summary, comparison tables, regression output
Stage 5 CI         pr profile, nightly, baseline comparison
```

Only then retrieval, analytical, graph, operations, and AI.

**Unresolved tension, do not paper over it:** TRD §31 was written when the vector suite was the primary target. Now that the agent workload is the primary surface (TRD §12), leading with Vector v1 means the first thing built is a component surface rather than the one the project exists to measure. Either the sequence gains an early agent smoke path, or the project knowingly accepts that its primary surface arrives late. This is an owner decision and has not been made.

The v0.1 definition of done (TRD §31) is operational, not subjective: a third party clones, runs `doctor`, fetches/verifies the dataset, bootstraps TheoDB plus one baseline, executes an ANN benchmark, and receives raw measurements, environment metadata, a validation status, and a report — **with no undocumented manual step**.

---

## Profiles

`smoke` (fast local validation, not publishable) · `pr` (regression gate on controlled hardware) · `nightly` (larger datasets, broad telemetry) · `release` (frozen methodology, raw artifacts retained, publishable) · `research` (explicitly non-authoritative).

GitHub-hosted shared runners are **not** authoritative for small performance deltas (TRD §23). Correctness/schema/unit tests run on the PR; real benchmarks are dispatched to a dedicated runner.

---

## Structure and extension points

The repository is greenfield: **the first structure laid down is the one that persists**. Treat it as a design decision, not an implementation detail.

**TRD §4 is the canonical layout** (`cmd/`, `runner/{orchestrator,launcher,isolation,lifecycle,validation}`, `telemetry/`, `analysis/`, `tests/`). PRD §15 and the README hold a shallower public view of the same thing; where they diverge, TRD §4 wins and the other two are corrected to match. Do not invent a third layout.

Folders are organized by capability (`bench/agent`, `bench/vector`, `telemetry/linux_perf`, `analysis/pareto`), not by generic technical layer — keep it that way, and do not create `utils/`, `helpers/`, `common/`, or `misc/`.

The runner is **Python**. `cmd/theodb-bench/` in TRD §4 reads as a Go convention; it is not one here — the CLI entry point follows Python packaging conventions, and TRD §4 should be corrected on that point when the first code lands.

The variation points are **real and already declared in the documents** — this is where OCP/DIP pay for themselves, and adding a new case must be an addition of code, never surgery on the core:

| Seam | Contract | Adding a new one must NOT require touching |
|---|---|---|
| System adapter | TRD §26 (`prepare…cleanup` + capabilities JSON) | any benchmark suite (D2) |
| Telemetry collector | TRD §18–19 (perf, process, io, postgres, latency) | the orchestrator |
| Workload/benchmark spec | TRD §3 D1 (declarative, versioned) | the runner core (PRD §16) |
| Report renderer | TRD §7 (`report/{markdown,json,charts}`) | orchestration (D7) |

Outside those seams, the moderation in `~/.claude/CLAUDE.md` applies: an interface with a single implementer and no foreseen variation is speculative abstraction, not state of the art. The bar is to decide consciously — neither abstracting by reflex nor letting the structure emerge by accident.

---

## Branching flow

```
workspace ──PR──> develop ──PR + semver tag──> main
 (work)          (integration)                (release)
```

- **`workspace`** is where work is born — single, permanent, never deleted per task. Every change commits here first.
- **`develop`** integrates and never originates. It advances only through a `workspace → develop` PR.
- **`main`** is release-only: a `develop → main` PR plus a semver tag on merge.

**This was absent until 2026-08-20, and the absence was the finding.** The repository had exactly one branch
(`workspace`), which was also the GitHub default — so every push went straight to the default with no PR gate,
while `.github/workflows/ci.yml` already declared `branches: [main, develop, workspace]` plus a `pull_request`
trigger. The flow was assumed by the CI and did not exist on the remote (`theo-db` BACKLOG, B-062).

Now in place, and verified with `gh` on 2026-08-20: `main` created from `develop`; the default branch moved to
`main`, because a default that is the working branch invites the accidental direct push; and branch protection
on **both** `main` and `develop` requiring a pull request, with `enforce_admins` on and force-push and deletion
denied.

`required_approving_review_count` is **0**, deliberately. GitHub does not let an author approve their own pull
request, so on a single-maintainer repository requiring one approval is a deadlock, not a gate. Zero still
requires the PR — which is the guarantee that was missing — while leaving the merge possible.

Two layers, two different guarantees, and they are not interchangeable (`rules/git-safety.md § 1`): the local
hook governs where work is **born**; branch protection is what makes the PR **mandatory**. This repository had
neither until now.

## Writing code here

- **The runner is a measurement instrument.** It must minimize the probability that its own orchestration behavior alters what is being measured (TRD §1). An expensive collector runs between repetitions or has its overhead measured (TRD §18).
- **The benchmark needs its own tests** (TRD §28): unit (schema, statistics, Pareto logic, checksums, latency histogram), integration (fake SUT, process-tree enforcement, timeout, crash, finalization), golden (deterministic report generation, known regression classification, known ANN quality calculation).
- **Large public datasets do not go into Git** — manifest + checksum + documented acquisition (TRD §25).
- **No untrusted contribution executes automatically on privileged release hardware** (TRD §24).
