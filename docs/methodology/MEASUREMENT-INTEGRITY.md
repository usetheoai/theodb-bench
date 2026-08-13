# Measurement integrity invariants

**Status:** normative · **Applies to:** every benchmark, adapter, and analysis module in this repository.

Each invariant below was discovered through a **real measurement defect**. None of them is a style preference or a matter of taste. Dropping one does not produce a benchmark that is merely less rigorous — it produces numbers that look correct and are not, and the failure is silent.

Read this before writing any code that measures, compares, or reports.

## Provenance and applicability

These invariants were established empirically over roughly a dozen milestones of a predecessor measurement harness, and are recorded here as the contract for TheoDB Bench. They are preserved in full.

Their scope is not uniform, and pretending otherwise would itself be a form of dishonesty:

- **I1–I9, I14–I23** state something true about measurement in general. They apply unchanged, whatever is being measured.
- **I10–I13** were derived from specific competitor index implementations (pgvector IVFFlat, pgvectorscale DiskANN) and from TheoDB's own access methods being L2-only. Their *letter* is competitor-specific; their *principle* — a comparison must not silently hand one system a crippled configuration — is general. When an adapter for a different system is added, restate the equivalent trap for that system rather than assuming these four cover it.

This repository benchmarks a database built for **agents**. That target changes what we measure, not what makes a measurement honest.

---

## 1. Recall

**I1. Recall is a distance threshold, not id overlap.**
Follow ANN-Benchmarks (Aumüller, Bernhardsson & Faithfull, arXiv:1807.05614 §2.1):

```
recall = |{returned with dist ≤ dist(k-th true neighbor) + eps}| / k
```

Id overlap diverges from the standard under tied or duplicated distances. Use `eps = 1e-3`.

**I2. The oracle sees float32.**
Vector columns are stored as float4. Ground truth must round to float32 **and then** compute the distance in float64. Without this, near-ties diverge between the oracle and the system under test.

**I3. At ≥1M scale, ground truth comes from `neighbors`, recomputed.**
Use the pre-computed neighbor ids from the dataset file and recompute the distances from the **vectors**. Never trust the `distances` array shipped in the file. This is 10⁶ operations instead of 10¹⁰.

**I4. Out-of-range neighbor ids fail loudly.**
NumPy would silently wrap an invalid index and produce wrong ground truth.

## 2. Executing the measurement

**I5. The index is forced *and* verified.**
`SET enable_seqscan = off` **plus** an assertion that reads the actual `EXPLAIN` output. Forcing without verifying proves nothing.

**I6. Specs are isolated from each other.**
Before measuring a spec, drop the indexes belonging to every other spec. Otherwise the planner chooses arbitrarily between two indexes of the same family on the same column, and one sweep flattens onto the other.

**I7. Warm-up is untimed.**
An untimed warm-up precedes the timed rounds, so that percentiles and throughput describe a consistently warm cache.

**I8. Throughput is `1 / min(per-round mean)` — best-of-N**, per the ANN-Benchmarks protocol. Any reported `mean`/`std` is **intra-sample** dispersion, not between-round variance, and the report must say so explicitly.

**I9. A capped sample appears in the label.**
An index that is O(N) per query may have its sample reduced — and the cap goes into the label (`[q=200]`), never hidden.

## 3. Fairness between compared systems

> See the applicability note above: I10–I13 are stated in terms of specific competitor indexes. Preserve the principle when adding a new system.

**I10. IVFFlat `lists` derives from the real N.**
When training on the full dataset, read the true size from the dataset file. Deriving from a default (5000 → `lists=5`) would build a crippled IVFFlat over 1M vectors — an unfair comparison wearing the costume of a measurement.

**I11. `probes` is clamped to `lists` BEFORE de-duplication.**
In pgvector, `probes > lists` is a no-op; a `probes=10` label on a `lists=5` index would report a duplicate point under the wrong label.

**I12. DiskANN `query_rescore` scales with `sls`** (up to pgvectorscale's ceiling of 1000). Freezing rescore below sls caps recall while throughput falls — fabricating a false plateau.

**I13. Never fabricate an opclass.**
Where an access method is L2-only, a run requesting cosine **warns and skips** rather than emitting invalid DDL.

## 4. Statistics

**I14. Comparing two systems requires a paired test** — not a comparison of means, and not a coefficient of variation. The headline test is a **paired randomization/permutation test** (Smucker, Allan & Carterette, CIKM 2007 — the recommended choice in IR). Wilcoxon and the sign test are rejected: they discard magnitude and ties.

**I15. Report three things:** a 95% CI from a paired percentile bootstrap, and a paired t-test as a concordant cross-check (Urbano et al., SIGIR 2013).

**I16. Monte-Carlo correction is mandatory:** `p = (count + 1) / (B + 1)`. The observed assignment is one of the permutations; `p` is never 0.

**I17. The seed is fixed**, so that `p` and the confidence interval reproduce exactly.

**I18. Report effect size, not just `p`:** `mean_diff`, `cohen's dz`, and `wins/losses/ties`.

## 5. Determinism and provenance

**I19. The corpus is seeded.** The same seed yields a bit-identical corpus and query set. The open-source analogues do not seed; this is a deliberate differentiator.

**I20. Deterministic tie-breaking on every ranking query** (`ORDER BY score, doc_id`). Without the id, ties at the top-N boundary are resolved by physical table order — non-deterministic across runs.

**I21. Every report carries provenance:** git `sha`, `date`, `seed`, `host`, `gt_source`, and a prose `methodology` field.

**I22. Regression comparison is byte-identical.** Differing qid sets are a **typed error**, never a silent partial comparison. No off-the-shelf suite (ann-benchmarks, VectorDBBench, ClickBench, BenchBase) offers this; it is our own capability, and it has caught real defects.

**I23. Model API keys are never logged, echoed, or persisted.** A missing key raises. A wrong dimension or an all-zero vector raises. There is no silent fallback to a zero vector — it would corrupt recall without warning.

---

## Relationship to the invariants in `CLAUDE.md`

`CLAUDE.md` states ten project-level invariants derived from the PRD and TRD — what may be published, what invalidates a run, how systems are compared. This document states what makes an individual **number** trustworthy. Both hold simultaneously; neither subsumes the other.
