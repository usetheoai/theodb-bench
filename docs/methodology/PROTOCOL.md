# Protocol

What a run does, in what order, and what makes it valid. This document is
normative: the runner implements it, and a run that deviates is invalid
regardless of what it measured.

## The eleven phases

| # | Phase | What must hold |
|---|---|---|
| 0 | Preflight | Mandatory checks for the profile pass. A failure stops the run **before** measurement, not after. |
| 1 | Environment capture | The host is recorded before the workload perturbs it. |
| 2 | Isolation | Declared controls are applied; what could not be applied is recorded as unenforced. |
| 3 | Bootstrap | The system starts, becomes ready, and reports the configuration actually in force. |
| 4 | Dataset load | Data is verified by checksum, then loaded. Load time is measured separately from query time unless load *is* the benchmark. |
| 5 | Index build | Measured separately. Build time, index size and the parameters in force are recorded. |
| 6 | Warm-up | Untimed, following a declared policy. |
| 7 | Measurement | The timed window. Telemetry starts before it and stops after it. |
| 8 | Cooldown / repetition | Repetition semantics are part of the benchmark definition, not of the operator's judgement. |
| 9 | Validation | Protocol checks over observed facts. |
| 10 | Finalization | The manifest is written last and the bundle is frozen. |

The manifest being written last is not bookkeeping: a bundle carrying one is a
bundle whose measurement completed.

## Warm-up

Warm-up must follow a policy declared in the benchmark definition — a fixed
time, a fixed operation count, or none.

**Warm-up may not be extended until a desired number appears.** This is the
single easiest way to produce a favourable result that looks honest, and the
validation check `warmup_policy` exists specifically to catch it.

## Repetition semantics

A benchmark declares what happens between repetitions:

- is the index rebuilt?
- is the system restarted?
- are caches dropped?
- is the database restored?
- is only the client restarted?

These are not tuning knobs. A suite that rebuilds the index between repetitions
measures something different from one that does not, and the two numbers are
not comparable.

## Validation

A run is `VALID`, `INVALID`, or `EXPLORATORY`.

Validation consults **only** how the run was executed. The record it operates
on carries no throughput, no latency and no recall — a property asserted
directly by a test — which is what makes it structurally impossible for a
favourable number to rescue a broken run, or an unfavourable one to condemn a
sound run.

Fifteen checks, each recorded with its outcome and whether it was mandatory for
the profile: system alive, client alive, operation count, repetitions
completed, timeout rate, error rate, result integrity, warm-up policy, process
containment, CPU limit, memory limit, no OOM, telemetry completeness, quality
reported, clean source tree.

An observation that could not be made is `UNAVAILABLE`, never `PASS`. Not
having looked is not the same as having looked and found nothing wrong.

## Absences

No metric is ever serialised as `0` when the truth is that it was not measured.
Four distinct reasons are recorded, and they are not interchangeable:

| Reason | Meaning |
|---|---|
| `unsupported` | The system cannot do this at all. |
| `unavailable` | Supported in principle, not obtainable here. |
| `not_collected` | Obtainable, but this run did not ask. |
| `invalid` | Collected, but failed validation. |

Zero cache misses is a finding. An unavailable counter is not.

## Unsupported features

A workload feature the system does not support produces an explicit
`unsupported` result. It is not a failed run, and it is never a substituted
measurement from a different code path.

## Profiles

| Profile | Min repetitions | Isolation | Preflight | Publishable |
|---|---|---|---|---|
| `smoke` | 1 | optional | optional | no |
| `pr` | 3 | required | required | no |
| `nightly` | 3 | required | required | no |
| `release` | 5 | required | required | **yes** |
| `research` | 1 | optional | optional | no |

Only `release` is publishable, and a `release` run additionally requires frozen
methodology, frozen datasets, complete telemetry and a clean source tree. A
research run is `EXPLORATORY` even when technically clean, because its
parameters are not frozen.

## Invalidation

**Invalidation is based on protocol criteria, never on the measured outcome.**

A run is not discarded for being unfavourable. No code path in this project
removes or filters a run based on the value of a metric, and none may be added.
