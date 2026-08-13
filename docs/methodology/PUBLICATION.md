# Publication

What it takes to turn a measurement into a claim.

## The chain

Every published number must be traceable:

```
claim -> report -> summary.json -> run manifest -> benchmark protocol
      -> source commit -> dataset checksum -> configuration -> raw measurements
```

A claim that cannot be walked back along this chain is not supported by this
framework, whatever number it quotes.

## Preconditions

A result is publishable only when **all** of these hold:

1. the profile is `release`;
2. the run status is `VALID`;
3. the benchmark version and commit are identified;
4. the system version and commit are identified;
5. the dataset identity is verified by checksum;
6. the environment capture is complete;
7. the warm-up and measurement policies were satisfied;
8. isolation checks passed;
9. raw measurements are retained;
10. statistical validation passed;
11. for a comparison, the fairness policy passed.

`summary.json` carries a `publishable` boolean computed from the first two, and
a `limitations` list. A run that fails any precondition is published as what it
is, or not published — never quietly promoted.

## What accompanies a published result

- the human report;
- the machine summary;
- the run manifest;
- the raw artifacts;
- the effective configuration of every system compared;
- the dataset manifest;
- the benchmark commit and the system commit;
- known limitations.

Raw measurements are part of the result. A Markdown table alone is not a
benchmark artifact.

## Corrections

**A published result is never silently overwritten.**

A correction creates a new report and marks the earlier one superseded, with
the reason. The superseded result stays reachable: readers who acted on it are
entitled to see what changed.

## Negative results

A result showing TheoDB slower is published under the same rules as one showing
it faster. Nothing in this framework delays, filters or re-runs a result
because of which way it went.

The success condition for this project is that someone can use it to prove
TheoDB is faster **or** slower under a clearly defined workload.

## What may not be claimed

- A component measurement is not a product claim. A faster kernel is evidence
  about that kernel.
- A `smoke`, `pr`, `nightly` or `research` number is not evidence, whatever it
  shows.
- Throughput without a quality axis is not an approximate-search result.
- A difference between two medians is not a significant difference; that
  requires the paired test in STATISTICS.md, run over per-query values.
- "TPC-H-derived" is not a TPC result.
