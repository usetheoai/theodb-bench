"""Preflight: deciding whether this host may measure anything.

A preflight failure stops a run before measurement rather than producing a
number nobody can trust (TRD 6.0). Which checks are mandatory depends on the
profile: a smoke run on a laptop is legitimate, the same host producing a
release claim is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from theodb_bench.command import which
from theodb_bench.environment import (
    capture_capabilities,
    capture_cpu,
    capture_memory,
    capture_software,
    capture_storage,
)
from theodb_bench.profiles import Profile


class Outcome(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class Check:
    """One preflight finding."""

    id: str
    outcome: Outcome
    detail: str
    required_for: frozenset[str] = frozenset()
    """Profiles for which this check failing must stop the run."""

    def is_blocking_for(self, profile: Profile) -> bool:
        """Whether this finding stops a run under the given profile.

        Anything short of PASS blocks when the profile declares the check
        mandatory. A warning is not automatically harmless: a release claim
        measured under frequency scaling, with swap enabled, or with NUMA
        placement uncontrolled is a methodology defect, not a note. Which
        checks are mandatory is the profile's decision, made in `required_for`.
        """
        return self.outcome is not Outcome.PASS and profile.name.value in self.required_for

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "required": sorted(self.required_for),
        }


_GATED: Final[frozenset[str]] = frozenset({"pr", "nightly", "release"})
_RELEASE_ONLY: Final[frozenset[str]] = frozenset({"release"})


def _absent_detail(value: Any, fallback: str) -> str:
    if isinstance(value, dict) and "absent" in value:
        detail = value.get("detail")
        return f"{value['absent']}: {detail}" if detail else str(value["absent"])
    return fallback


def check_operating_system() -> Check:
    software = capture_software()
    os_name = software["os"]
    kernel = software["kernel"]
    if isinstance(os_name, dict) or isinstance(kernel, dict):
        return Check(
            "operating_system",
            Outcome.UNAVAILABLE,
            _absent_detail(os_name, "OS could not be identified"),
            _GATED,
        )
    return Check("operating_system", Outcome.PASS, f"{os_name}, kernel {kernel}")


def check_cpu() -> Check:
    cpu = capture_cpu()
    logical = cpu["logical_cpus"]
    if not isinstance(logical, int):
        return Check("cpu", Outcome.UNAVAILABLE, _absent_detail(logical, "no CPU count"), _GATED)
    physical = cpu["physical_cores"]
    cores = physical if isinstance(physical, int) else "unknown"
    if logical < 4:
        return Check(
            "cpu",
            Outcome.WARN,
            f"{logical} logical CPUs; too few to separate client from system under test",
            _GATED,
        )
    return Check("cpu", Outcome.PASS, f"{logical} logical CPUs, {cores} physical cores")


def check_smt() -> Check:
    smt = capture_cpu()["smt_enabled"]
    if not isinstance(smt, bool):
        return Check("smt", Outcome.UNAVAILABLE, _absent_detail(smt, "SMT state unknown"))
    if smt:
        # Not a failure: SMT is a legitimate configuration. It is a warning
        # because sibling threads make per-core measurements noisier, and a
        # release claim should say which state it was measured in.
        return Check("smt", Outcome.WARN, "SMT enabled; sibling threads add measurement noise")
    return Check("smt", Outcome.PASS, "SMT disabled")


def check_cpu_governor() -> Check:
    governor = capture_cpu()["frequency_policy"]
    if not isinstance(governor, str):
        return Check(
            "cpu_governor",
            Outcome.UNAVAILABLE,
            _absent_detail(governor, "cpufreq not exposed"),
            _RELEASE_ONLY,
        )
    if governor != "performance":
        return Check(
            "cpu_governor",
            Outcome.WARN,
            f"governor is {governor!r}; frequency scaling varies results between runs",
            _RELEASE_ONLY,
        )
    return Check("cpu_governor", Outcome.PASS, "governor is 'performance'")


def check_memory() -> Check:
    memory = capture_memory()
    total = memory["total_bytes"]
    if not isinstance(total, int):
        return Check(
            "memory", Outcome.UNAVAILABLE, _absent_detail(total, "no memory total"), _GATED
        )
    gib = total / (1024**3)
    return Check("memory", Outcome.PASS, f"{gib:.1f} GiB total")


def check_swap() -> Check:
    swap = capture_memory()["swap_total_bytes"]
    if not isinstance(swap, int):
        return Check("swap", Outcome.UNAVAILABLE, _absent_detail(swap, "swap state unknown"))
    if swap > 0:
        return Check(
            "swap",
            Outcome.WARN,
            f"{swap / (1024**3):.1f} GiB swap enabled; paging silently distorts latency",
            _RELEASE_ONLY,
        )
    return Check("swap", Outcome.PASS, "swap disabled")


def check_storage() -> Check:
    devices = capture_storage()
    if not devices:
        return Check("storage", Outcome.UNAVAILABLE, "no real block devices found", _GATED)
    described = ", ".join(f"{d['device']} ({d['filesystem']})" for d in devices[:3])
    return Check("storage", Outcome.PASS, f"{len(devices)} device(s): {described}")


def check_perf_events() -> Check:
    perf = capture_capabilities()["perf_events"]
    if not isinstance(perf, bool):
        return Check("perf_events", Outcome.UNAVAILABLE, _absent_detail(perf, "perf unknown"))
    if not perf:
        # Not fatal anywhere: hardware counters are optional telemetry, and the
        # result records them as unavailable rather than as zero.
        return Check(
            "perf_events",
            Outcome.WARN,
            "hardware counters unavailable; cycles and cache misses will be recorded as absent",
        )
    return Check("perf_events", Outcome.PASS, "perf events accessible")


def check_cgroups() -> Check:
    cgroup_v2 = capture_capabilities()["cgroup_v2"]
    if not isinstance(cgroup_v2, bool):
        return Check("cgroup_v2", Outcome.UNAVAILABLE, _absent_detail(cgroup_v2, "unknown"), _GATED)
    if not cgroup_v2:
        return Check(
            "cgroup_v2",
            Outcome.FAIL,
            "cgroup v2 not mounted; memory limits and process-tree containment cannot be enforced",
            _GATED,
        )
    return Check("cgroup_v2", Outcome.PASS, "cgroup v2 available")


def check_cpu_affinity() -> Check:
    affinity = capture_capabilities()["cpu_affinity"]
    if not isinstance(affinity, bool) or not affinity:
        return Check(
            "cpu_affinity",
            Outcome.FAIL if affinity is False else Outcome.UNAVAILABLE,
            "CPU affinity cannot be set; declared CPU sets would not be enforced",
            _GATED,
        )
    return Check("cpu_affinity", Outcome.PASS, "CPU affinity supported")


def check_numa() -> Check:
    numa_nodes = capture_cpu()["numa_nodes"]
    control = capture_capabilities()["numa_control"]
    if isinstance(numa_nodes, int) and numa_nodes <= 1:
        return Check("numa", Outcome.PASS, "single NUMA node; placement is not a variable")
    if control is not True:
        return Check(
            "numa",
            Outcome.WARN,
            "multiple NUMA nodes but numactl is absent; placement will be uncontrolled",
            _RELEASE_ONLY,
        )
    return Check("numa", Outcome.PASS, "NUMA placement controllable")


def check_required_binaries() -> Check:
    # git is how a run records which source produced it; without it a bundle
    # cannot carry provenance.
    missing = [name for name in ("git",) if which(name) is None]
    if missing:
        return Check(
            "required_binaries",
            Outcome.FAIL,
            f"missing: {', '.join(missing)}",
            frozenset({"smoke", "pr", "nightly", "release", "research"}),
        )
    return Check("required_binaries", Outcome.PASS, "git present")


def check_postgres_client() -> Check:
    version = capture_software()["postgres"]
    if not isinstance(version, str):
        return Check(
            "postgres_client",
            Outcome.UNAVAILABLE,
            _absent_detail(version, "psql not found"),
        )
    return Check("postgres_client", Outcome.PASS, version)


def check_container_runtime() -> Check:
    runtime = capture_software()["container_runtime"]
    if not isinstance(runtime, str):
        return Check(
            "container_runtime",
            Outcome.UNAVAILABLE,
            _absent_detail(runtime, "no container runtime found"),
        )
    return Check("container_runtime", Outcome.PASS, runtime)


def check_datasets(dataset_root: Path) -> Check:
    if not dataset_root.exists():
        return Check(
            "datasets",
            Outcome.UNAVAILABLE,
            f"{dataset_root} does not exist; nothing fetched yet",
        )
    present = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
    if not present:
        return Check("datasets", Outcome.UNAVAILABLE, f"{dataset_root} is empty")
    return Check("datasets", Outcome.PASS, f"{len(present)} dataset(s): {', '.join(present)}")


CHECKS: Final[tuple[Callable[[], Check], ...]] = (
    check_operating_system,
    check_cpu,
    check_smt,
    check_cpu_governor,
    check_memory,
    check_swap,
    check_storage,
    check_perf_events,
    check_cgroups,
    check_cpu_affinity,
    check_numa,
    check_required_binaries,
    check_postgres_client,
    check_container_runtime,
)


@dataclass(frozen=True)
class DoctorReport:
    """Every check, plus what they mean for a given profile."""

    checks: tuple[Check, ...]
    profile: Profile

    @property
    def blocking(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.is_blocking_for(self.profile))

    @property
    def may_run(self) -> bool:
        """Whether a run under this profile may start."""
        return not self.blocking

    def counts(self) -> dict[str, int]:
        tally = {outcome.value: 0 for outcome in Outcome}
        for check in self.checks:
            tally[check.outcome.value] += 1
        return tally

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name.value,
            "may_run": self.may_run,
            "counts": self.counts(),
            "blocking": [c.id for c in self.blocking],
            "checks": [c.as_dict() for c in self.checks],
        }


def run_doctor(profile: Profile, dataset_root: Path | None = None) -> DoctorReport:
    """Execute every preflight check and judge the result for a profile."""
    checks = [check() for check in CHECKS]
    if dataset_root is not None:
        checks.append(check_datasets(dataset_root))
    return DoctorReport(tuple(checks), profile)


def render_report(report: DoctorReport) -> str:
    """Human-readable doctor output."""
    symbol = {
        Outcome.PASS: "PASS",
        Outcome.WARN: "WARN",
        Outcome.FAIL: "FAIL",
        Outcome.UNAVAILABLE: "N/A ",
    }
    width = max(len(c.id) for c in report.checks)
    lines = [f"theodb-bench doctor  (profile: {report.profile.name.value})", ""]
    for check in report.checks:
        marker = "*" if check.is_blocking_for(report.profile) else " "
        lines.append(f"{symbol[check.outcome]}{marker} {check.id.ljust(width)}  {check.detail}")
    tally = report.counts()
    lines.append("")
    lines.append(
        f"{tally['PASS']} pass, {tally['WARN']} warn, "
        f"{tally['FAIL']} fail, {tally['UNAVAILABLE']} unavailable"
    )
    if report.may_run:
        lines.append(f"Host may run a '{report.profile.name.value}' benchmark.")
    else:
        blocking = ", ".join(c.id for c in report.blocking)
        lines.append(
            f"Host may NOT run a '{report.profile.name.value}' benchmark. Blocking: {blocking}"
        )
        lines.append("Checks marked * are mandatory for this profile.")
    return "\n".join(lines)


__all__ = [
    "Check",
    "DoctorReport",
    "Outcome",
    "render_report",
    "run_doctor",
]
