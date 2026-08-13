# Fairness

Rules for comparing systems. The purpose is not to be generous to competitors;
it is to make a comparison mean something. A benchmark that is unfair is not
merely impolite — it is wrong, and its numbers describe the configuration
rather than the system.

## Equal rules

Any rule applied to TheoDB applies to every compared system, unless the
benchmark documents why a capability-specific exception is required.

This covers, at minimum: CPU allocation, memory allocation, storage, durability
settings, dataset, preprocessing, load procedure, cache state, warm-up,
concurrency, measurement window, repetitions and client placement.

TheoDB-specific tuning is permitted **only** when equivalent system-specific
tuning is permitted for the competitor under the same published policy.

## Effective configuration, not requested configuration

Every adapter exports the configuration actually in force, read back from the
running system. What was requested is not evidence that it took effect.

A comparative report must show material differences between the systems. A
comparison where one side had four times the cache is a legitimate measurement
of a configuration and an illegitimate claim about an engine.

## Approximate indexes

An approximate index trades quality for speed. Comparing throughput without
quality is therefore not a comparison at all — the faster system may simply be
returning worse answers.

Two acceptable forms:

1. **Matched quality.** State the target and the selection method. The
   implementation refuses to promote a near miss: 0.949 is not 0.95, and
   reporting it as if it were is how a matched-recall comparison stops being
   matched.
2. **The complete Pareto frontier.** Every non-dominated configuration, so the
   reader picks the operating point.

## Not crippling the competitor

An unfair comparison usually arrives as an innocent default, not as malice.
The cases already encountered, and enforced in code:

- **IVFFlat `lists` derives from the real row count.** Sizing a million-row
  index from a default row count builds a crippled index and calls the result a
  comparison.
- **`probes` is clamped to `lists`.** Above `lists` it is a no-op, and the
  point would appear in the table under a label that does not describe it.
- **Rescore depth scales with search depth.** Freezing it caps recall while
  throughput falls, fabricating a plateau that does not exist.
- **A missing operator class is `unsupported`.** Emitting DDL for an opclass
  that does not exist turns a missing capability into a failed run, which reads
  as a defect rather than as a gap.

When a new system is added, the equivalent traps for that system must be
identified and stated. These four are not a checklist that covers the next
adapter.

## Index use is verified, not assumed

`SET enable_seqscan = off` proves nothing on its own — the planner may still
choose a different path. Index use is asserted from the actual `EXPLAIN`
output, because a benchmark that assumed otherwise would report sequential scan
performance under an index's name.

Indexes belonging to other configurations are dropped before a point is
measured. Two indexes of the same family on the same column let the system
choose between them, and one sweep silently flattens onto the other.

## Attribution

Latency belongs to whoever caused it. In any pipeline involving an external
model, database time, network time and inference time are separated in the
report. A composite that silently includes inference measures the model vendor.

## Benchmark names

TPC workloads appear only as "TPC-H-derived" and never in a form implying an
audited TPC result. Where a tool's own name already makes this distinction —
HammerDB's TPROC-C, for instance — that name is used as the tool publishes it.

## Publishing losses

A valid result stays valid when TheoDB loses. Results are not filtered,
delayed, or re-run until they improve. If a benchmark exposes an architectural
limit, that is the benchmark working.
