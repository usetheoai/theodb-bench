# CI

Two classes, and the split is the point.

## Shared CI (`.github/workflows/ci.yml`)

Runs on every push and pull request. Formats, lints, typechecks, runs the test
suite, validates every schema fixture, and executes a smoke benchmark against
the fake system to prove the pipeline works end to end.

**Nothing it measures is authoritative.** GitHub-hosted runners are shared, of
unknown hardware class, and noisy. A timing difference observed there describes
the runner. The smoke benchmark exists to catch a broken pipeline, and its
numbers are discarded.

## Dedicated benchmark runner (`.github/workflows/benchmark.yml`)

Manual dispatch and a nightly schedule, on a self-hosted machine labelled
`benchmark`. Concurrency is pinned to one run at a time, because two benchmarks
sharing a host measure each other.

It does **not** trigger on `pull_request`: an untrusted contribution must never
execute automatically on privileged benchmark hardware.

## Before a gate can fail a build

`theodb-bench` will not fail a build on a threshold that was not derived from a
measured noise floor. Run the same benchmark repeatedly on the benchmark host
first, compute the coefficient of variation per metric, and derive gates from
it. Until then every gate reports ADVISORY and says why.
