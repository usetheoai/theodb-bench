# Statistics

How measurements become numbers, and what those numbers may claim.

## Every repetition is retained

Aggregates carry the values they came from. A reader sees the spread rather
than trusting a median, and `derived/statistics.json` contains the full list
per metric per configuration.

## No silent outlier removal

The default outlier policy is `none`. Any other policy must be named, versioned
and recorded in the artifact, together with every excluded value and the reason
for its exclusion.

A run is never removed because it was unfavourable. This is not a matter of
discipline; there is no code path that filters on a metric's value.

## Latency

Percentiles are computed over **successful** operations only. Timeouts and
errors are counted separately and never enter the distribution — folding them
in would make a system look faster the more often it failed.

`p99.9` is withheld below a thousand samples and reported as
`unavailable: p99.9 needs >= 1000 samples`. Below that it is the maximum
wearing a percentile's name.

## Throughput

Best-of-N, following the ANN-Benchmarks protocol: the reciprocal of the fastest
per-round mean. Mixing this with an average-of-rounds convention makes numbers
incomparable with every existing ANN result.

The dispersion reported alongside a throughput figure is within-sample, not
between-round, and the artifact says so.

## Confidence intervals

A normal approximation, and the code says it is one. With the handful of
repetitions a benchmark runs, a t-interval would be false precision. A single
repetition yields no interval at all — reported as an absence, not as zero
width.

## Instability

A configuration whose repetitions disagree by more than the declared
coefficient-of-variation threshold is flagged `unstable`.

Instability is **reported, never corrected**. The point stays in the result and
in the report; what changes is that the reader is told the median is a weaker
claim than it looks.

## Noise floor

A regression threshold must be derived from the benchmark's own measured
variance on the hardware in question. Run the same benchmark repeatedly, take
the coefficient of variation per metric, and derive budgets from that.

Until a noise floor exists, every gate reports `ADVISORY` and states that its
threshold was not derived from one. A gate tighter than the noise floor
produces alerts about the machine.

## Comparing two systems

A difference between two medians is an observation, not a result. A comparative
claim requires a paired test — the recommended choice in IR is a paired
randomisation test (Smucker, Allan & Carterette, CIKM 2007), reported alongside
a paired bootstrap confidence interval and a paired t-test as a concordant
cross-check (Urbano et al., SIGIR 2013).

Two properties are mandatory when that test is implemented:

- **Monte-Carlo correction:** `p = (count + 1) / (B + 1)`. The observed
  assignment is one of the permutations, so `p` is never 0.
- **Effect size, not only `p`:** mean difference, Cohen's dz, and
  wins/losses/ties.

> **Status.** Implemented in `theodb_bench.analysis.significance`. SciPy is
> optional: without it the t-test cross-check uses a normal approximation, and
> the artifact records which method produced the p-value rather than leaving a
> reader to assume the exact distribution was used.
