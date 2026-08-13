# 1. External load generators: pgbench now, transactional baseline deferred

**Status:** accepted (with one open question) · **Date:** 2026-08-12

## Context

TheoDB Bench measures a database built for agents; the unit of measurement is the agent step, not the query (TRD §12). Two mature external tools were proposed for adoption:

- **pgbench** — ships with PostgreSQL, reports TPS, implements a workload loosely based on TPC-B.
- **HammerDB** — reports TPM, implements TPROC-C, a TPC-C-derived OLTP workload.

Both are well maintained, widely used, and satisfy the "do not reinvent the wheel" rule. Neither measures anything on the primary surface: context assembly, filtered retrieval, memory staleness, concurrent agents, or task success.

The risk in adopting them is not technical, it is one of drift. Making transactional throughput a headline suite would re-import the framing of a classical database benchmark into a project that exists because that framing misses what an agent exercises.

## Decision drivers

- The regression gate cannot be frozen before the benchmark host's natural variance is measured (TRD §22, PRD §12). That measurement needs a standard, cheap, reproducible load.
- Closed-loop and open-loop workloads must be distinguished, and coordinated omission accounted for (TRD §11). A reference implementation to validate our own generator against is worth more than another suite.
- The agent write-back path runs continuously against the transactional engine. The claim that custom access methods, the columnar TAM, and the background vectorizer do not degrade it needs a transactional baseline (`read_write_interference_delta`, TRD §12.3).
- A component result is not a product claim (PRD P6).

## Decision

**Adopt pgbench now, as an instrument — not as a source of published performance claims.** Its roles:

1. Characterizing the noise floor of the benchmark host, which gates the entire regression model.
2. Preflight load check in `doctor`, so a host is rejected before spending a full measurement window.
3. An open-loop reference (`--rate`, latency under load) against which our own load generator is validated.

**Do not publish pgbench TPS as a comparative claim.** TPC-B was retired by the TPC, and pgbench's own documentation describes its workload as only loosely based on it. A pgbench number supports no statement about TheoDB.

**Adopt a transactional baseline later**, scoped to read/write interference, sequenced with the operations surface rather than with the runner core. Which tool fills that slot is deferred — see the open question.

**Neither tool becomes a headline surface.** At most they are component surfaces that explain an agent result.

## Open question: HammerDB or BenchBase

For the transactional-baseline slot, HammerDB is not the only candidate:

| | HammerDB | BenchBase (CMU) |
|---|---|---|
| Transactional workload | TPROC-C (TPC-C-derived) | TPC-C |
| HTAP in one tool | no | yes — CH-benCHmark combines TPC-C and TPC-H on one instance |
| Scripting surface | Tcl | Java, configuration-driven |
| License | believed GPL-3.0 — **must be verified** | believed Apache-2.0 — **must be verified** |

If the question being asked is interference between transactional and analytical work — which is the HTAP question, and closer to what an agent workload stresses — CH-benCHmark covers it more directly than TPROC-C alone.

Neither license claim above has been verified in this repository. Verification through the dependency license gate is a precondition of adoption. Because both tools would be invoked as external processes rather than linked, a copyleft license is not expected to be contaminating, but that reasoning must be recorded rather than assumed.

## Consequences

- The regression model becomes buildable: the noise floor has a defined measurement path.
- `doctor` gains a real preflight rather than a static capability check.
- Our load generator can be validated against a reference implementation instead of being trusted.
- One additional tool dependency in the `smoke`/`pr` path, dev-only, never shipped in an image.
- TPM and TPS remain absent from any TheoDB claim. If either ever appears in a report, it appears as a component result with that status stated.
- HammerDB naming helps rather than hinders compliance: TPROC-C is explicitly not audited TPC-C, matching the prohibition on official TPC branding (TRD §15, PRD §9.3).

## More information

- Primary surface and why transactional throughput is not it: `docs/methodology/AGENT-WORKLOAD.md`
- Measurement traps that apply to any generator adopted here: `docs/methodology/MEASUREMENT-INTEGRITY.md`
