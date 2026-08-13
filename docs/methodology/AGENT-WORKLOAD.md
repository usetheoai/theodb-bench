# Agent workload — the primary benchmark surface

**Status:** draft · **Level:** B3 (workload) and B4 (competitive)

TheoDB is built for agents. The benchmark's primary surface must therefore measure what an agent actually exercises, and the seven component surfaces (vector ANN, retrieval, AI SQL, graph, analytical, lakehouse, operations) become the explanations for why this surface moves.

## 1. Why the classical surfaces are not enough

Every widely used database benchmark measures a **query**. An agent does not issue a query; it takes a **step**. The two differ in ways that change which number matters:

| Agent property | What a query-centric benchmark reports instead |
|---|---|
| One step issues several retrievals (vector + lexical + graph) and completes only when the slowest returns | throughput of a single isolated query |
| Steps are serial within an agent, so the tail dominates the whole trajectory | mean or p50 latency |
| The agent writes memory continuously while reading it | a read-only workload over an index built once |
| Retrieval is filtered by agent, session, or tenant in essentially every call | unfiltered top-k over one hot index |
| Many agents hold small working sets; nothing stays warm | one large index, fully cached |
| What was written at step *N* must be retrievable at step *N+1* | freshness measured, if at all, in isolation from the loop |
| The question is whether the task completed | nDCG, recall — proxies for the question |

None of ANN-Benchmarks, VectorDBBench, BEIR, ClickBench, or BenchBase measures any row on the left. That gap is the reason this surface exists.

## 2. The unit of measurement: the agent step

A step is the smallest complete cycle an agent performs:

```
  context assembly            model call            write-back
 ┌────────────────────┐      ┌──────────┐      ┌──────────────────┐
 │ vector retrieval   │      │          │      │ observations     │
 │ lexical retrieval  │ ───► │   LLM    │ ───► │ derived memory   │
 │ graph expansion    │      │          │      │ (re-embedded)    │
 │ (filtered, tenant) │      └──────────┘      └──────────────────┘
 └────────────────────┘
        ▲                                               │
        └───────────────────────────────────────────────┘
                     next step must see it
```

The database is exercised in **context assembly** and **write-back**. The model call is not ours to measure, and attributing its latency to the database is the single most common way an agent benchmark lies (TRD §13, §17).

**The headline metric is step assembly latency: the p95/p99 of the composite of every database call within one step** — not the throughput of any individual leg. A system can win on per-query QPS and lose the step, and that inversion is exactly what we need to be able to show.

## 3. What is measured

### 3.1 Deterministic, gate-able (no model in the loop)

Measured with the `mock` endpoint (TRD §17), so results are reproducible and may back regression gates:

- **Step assembly latency** — p50/p95/p99/p99.9 of the composite; plus per-leg breakdown, so a regression is attributable.
- **Filtered retrieval quality and cost** — recall and latency across a filter-selectivity sweep. Per-tenant filtering is the default case, not an extra: graph-based indexes degrade sharply as selectivity drops, and pre-filter versus post-filter strategy is the whole ballgame. A benchmark that only measures unfiltered search does not describe agent usage.
- **Read-your-writes staleness** — for an observation written at step *N*, the probability it is retrievable at step *N+k*, and the staleness window distribution. See §4.
- **Concurrent agent scaling** — *M* agents with small, disjoint working sets against one instance. Measures cache behavior and filter cost under realistic fragmentation, not one hot index.
- **Read/write interference** — assembly latency while write-back and background vectorization run, versus the same measurement quiesced. The delta is the number; reporting either half alone is misleading.

### 3.2 Model-dependent, never a hard gate

Measured with a `local` frozen model, and marked environment-dependent:

- **Task success rate** — did the agent complete the task, given what the database returned.
- **Cost per step** — tokens plus database CPU.

These may never back an automatic performance gate (TRD §17). They exist because retrieval quality that does not change task outcome is a metric optimizing itself.

## 4. Read-your-writes is a correctness property, not a slow number

If an agent writes an observation and cannot retrieve it on the next step, the system did not perform slowly — it lost data the agent believed it had stored. A run that violates the declared staleness bound is therefore **INVALID** under the protocol (TRD §6 phase 9), not a poor result.

The bound must be declared by the benchmark spec, per pipeline. "Eventually retrievable" is not a bound.

## 5. Methodological traps specific to this surface

Additions to `MEASUREMENT-INTEGRITY.md`, in the same spirit: each of these produces a plausible wrong number.

**A1. Never report a step latency that includes model time.** Separate database, network, and inference in every report. A composite that silently includes inference measures the model vendor.

**A2. Compose the step from real concurrency, not from a sum of parts.** Measuring each leg alone and adding the results reports a step that never happened — it ignores both intra-step parallelism and contention.

**A3. Warm-up must not warm what the workload will not have warm.** Many agents with small working sets is a cold-ish regime by construction. A warm-up that loads every tenant's data into cache measures a system that will not exist in production.

**A4. Filter selectivity is a swept axis, never a single point.** One selectivity value hides the exact place where a graph index collapses.

**A5. Tenant assignment is seeded and reported.** Which agent touches which partition determines cache behavior; unseeded assignment makes runs incomparable.

**A6. A step that failed is not a step that was fast.** Timeouts, empty retrievals, and constraint violations are counted and reported separately, never folded into the latency distribution as if they had succeeded.

**A7. The mock endpoint must have a declared, non-zero latency profile.** A zero-latency model changes the concurrency regime of the whole loop and would flatter any system that overlaps I/O with inference.

## 6. Relationship to the component surfaces

The seven component surfaces are not demoted; they are what makes this one diagnosable.

| When the agent surface regresses | Look at |
|---|---|
| assembly p99 up, vector leg dominant | vector ANN suite |
| assembly p99 up, fusion leg dominant | retrieval suite |
| graph expansion leg dominant | graph suite |
| staleness window widened | operations / vectorizer suite |
| interference delta grew | operations suite |
| task success down at equal latency | retrieval quality, then AI SQL |

A component result explains an agent result. It does not substitute for one (PRD P6).

## 7. Open questions

These must be settled before this surface can produce publishable results:

1. Which agent trajectories constitute the reference workload, and where do they come from — recorded from real usage, synthesized from a task set, or derived from a public agent benchmark with a compatible license.
2. The memory schema an agent is assumed to use, since it determines the filter shape and therefore the numbers.
3. The declared staleness bound per pipeline (§4).
4. The mock endpoint's latency profile (A7).
5. Whether competitive comparison at this level is meaningful before the component surfaces are stable, or whether B4 for agents waits.
