"""The run orchestrator: the eleven phases of TRD section 6.

The orchestrator decides *what* happens and in which order. It never decides
*how* a system does anything -- that is the adapter's job -- and it never
decides whether a result is good, which is nobody's job.

Ordering is the substance, not bookkeeping. Preflight stops a run before
measurement rather than after; environment capture happens before the system
starts, so it describes the host rather than the host plus the workload; the
manifest is written last, so a bundle carrying one is a bundle whose
measurement completed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from theodb_bench.abort import AbortKind, classify_abort
from theodb_bench.absent import Absent
from theodb_bench.adapters.base import SystemAdapter
from theodb_bench.analysis.statistics import (
    PointStatistics,
    statistics_payload,
    summarise_points,
)
from theodb_bench.bench.protocol import Workload
from theodb_bench.bench.vector import PointResult
from theodb_bench.bundle import RunBundle
from theodb_bench.doctor import run_doctor
from theodb_bench.environment import capture_environment
from theodb_bench.errors import (
    ConfigError,
    ErrorContext,
    Phase,
    PreflightError,
)
from theodb_bench.isolation import (
    AppliedIsolation,
    IsolationPlan,
    apply_isolation,
    find_escapes,
    online_cpus,
)
from theodb_bench.profiles import Profile, get_profile
from theodb_bench.telemetry import CollectorSet, PerfStatCollector, ProcessCollector
from theodb_bench.validation import RunObservations, validate_run

BENCHMARK_SCHEMA_VERSION: Final[int] = 1
RESULT_SCHEMA_VERSION: Final[int] = 1

AdapterFactory = Callable[[], SystemAdapter]


@dataclass
class RunRequest:
    """Everything a run needs, decided before anything is measured."""

    benchmark_id: str
    workload: Workload
    adapter_factory: AdapterFactory
    profile: Profile = field(default_factory=lambda: get_profile("smoke"))
    benchmark_version: int = 1
    repetitions: int = 1
    results_root: Path = Path("results")
    isolation: IsolationPlan = field(default_factory=IsolationPlan)
    collect_process_telemetry: bool = True
    collect_perf_telemetry: bool = False
    dataset_id: str | None = None
    dataset_version: str | None = None
    dataset_sha256: str | None = None
    corpus: Any = None
    """Vectors from a verified dataset. Required whenever a dataset identity is
    declared: recording an id while measuring generated data would put a false
    provenance claim into an immutable bundle."""

    queries: Any = None


@dataclass
class RunOutcome:
    """What a completed run produced."""

    bundle: RunBundle
    status: str
    points: list[PointResult]
    statistics: list[PointStatistics]
    validation: dict[str, Any]
    environment: dict[str, Any]
    isolation: AppliedIsolation
    telemetry: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.bundle.run_id


def _benchmark_payload(request: RunRequest) -> dict[str, Any]:
    workload = request.workload
    warmup = workload.warmup_operations
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "id": request.benchmark_id,
        "version": request.benchmark_version,
        **workload.benchmark_payload(),
        "warmup": {
            "policy": "fixed_operations" if warmup else "none",
            "operations": warmup,
        },
        "repetitions": request.repetitions,
    }


def _result_payload(run_id: str, points: list[PointResult]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "points": [
            {
                "label": point.label,
                "parameters": point.parameters,
                "status": point.status,
                **({"status_detail": point.status_detail} if point.status_detail else {}),
                "repetitions": [
                    {
                        "repetition": repetition.repetition,
                        "operations": {
                            "success": repetition.successes,
                            "error": repetition.errors,
                            "timeout": repetition.timeouts,
                        },
                        "duration_seconds": repetition.duration_seconds,
                        **(
                            {"throughput_per_second": repetition.throughput}
                            if repetition.throughput is not None
                            else {}
                        ),
                        "latency_ms": repetition.latency.as_dict(),
                        **(
                            {
                                "quality": {
                                    f"recall_at_{point.parameters.get('k', '')}".rstrip(
                                        "_"
                                    ): repetition.recall
                                }
                            }
                            if repetition.recall is not None
                            else {}
                        ),
                        "resources": {
                            **(
                                {"build_seconds": repetition.build_seconds}
                                if repetition.build_seconds is not None
                                else {}
                            ),
                            **(
                                {"index_size_bytes": repetition.index_size_bytes}
                                if repetition.index_size_bytes is not None
                                else {}
                            ),
                        },
                    }
                    for repetition in point.repetitions
                ],
            }
            for point in points
        ],
    }


def _check_dataset_declaration(request: RunRequest) -> None:
    """A declared dataset identity must be the data that was actually measured.

    Without this the manifest could say `sift1m` while the run measured a
    seeded synthetic corpus, and nothing downstream could tell. Every artifact
    in the bundle would look correct; the provenance chain the whole project
    exists to provide would be broken at its first link.
    """
    declared = request.dataset_id is not None
    supplied = request.corpus is not None
    if declared and not supplied:
        raise ConfigError(
            f"run declares dataset {request.dataset_id!r} but supplies no vectors; "
            "a manifest may not name a dataset the run did not measure",
            context=ErrorContext(
                phase=Phase.PREFLIGHT,
                benchmark=request.benchmark_id,
                details={"dataset": request.dataset_id},
            ),
        )
    if supplied and not declared:
        raise ConfigError(
            "run supplies vectors but declares no dataset identity; measured data "
            "must be identifiable in the manifest",
            context=ErrorContext(phase=Phase.PREFLIGHT, benchmark=request.benchmark_id),
        )


def run_benchmark(request: RunRequest) -> RunOutcome:
    """Execute the full lifecycle and return an immutable bundle."""
    profile = request.profile
    profile.require_repetitions(request.repetitions)
    _check_dataset_declaration(request)

    # Phase 0 -- preflight. Before anything is measured, and before the system
    # is even started.
    doctor = run_doctor(profile)
    if profile.preflight_required and not doctor.may_run:
        blocking = ", ".join(check.id for check in doctor.blocking)
        raise PreflightError(
            f"host may not run a {profile.name.value!r} benchmark; blocking checks: {blocking}",
            context=ErrorContext(phase=Phase.PREFLIGHT, benchmark=request.benchmark_id),
        )

    # Phase 1 -- environment. Captured before the workload perturbs the host.
    environment = capture_environment()

    # Phase 2 -- isolation. What cannot be enforced is recorded as unenforced.
    applied = apply_isolation(request.isolation)

    adapter = request.adapter_factory()
    bundle = RunBundle.create(
        request.results_root,
        benchmark_id=request.benchmark_id,
        system_id=adapter.system_id,
    )
    bundle.write_artifact("environment", environment)
    bundle.write_artifact("benchmark", _benchmark_payload(request))

    # The workload builds its own benchmark. The orchestrator names no concrete
    # family, so a second one is a module rather than an edit here.
    benchmark = request.workload.build(request.corpus, request.queries)
    collectors = _build_collectors(request)

    abort_kind: AbortKind | None = None
    oom = False
    points: list[PointResult] = []
    try:
        # Phase 3 -- bootstrap.
        adapter.prepare()
        adapter.start()
        adapter.wait_ready()

        # Phase 4 -- dataset load, measured separately from the query benchmark.
        load_seconds = benchmark.load(adapter)
        bundle.write_raw_text("load.log", f"load_seconds={load_seconds:.6f}\n")

        collectors.start()
        # Phases 5 to 8 -- build, warm-up, measurement and repetition, per
        # configuration. Warm-up happens inside run_point and is untimed.
        points.extend(benchmark.points(adapter, request.repetitions))
        collectors.stop()

        bundle.write_artifact("system", adapter.system_payload())
        bundle.write_raw_text("system-stats.json", _as_json(adapter.collect_stats()))
    except Exception as exc:
        # Classified rather than blamed on the system under test. A gate refusing
        # to measure, a statement the harness itself cancelled, and a backend that
        # died are three different facts, and this report is published.
        abort_kind = classify_abort(exc)
        oom = bool(getattr(exc, "context", None) and exc.context.details.get("oom"))  # type: ignore[attr-defined]
        bundle.write_raw_text(
            "system.log",
            f"run aborted ({abort_kind.value}): {type(exc).__name__}: {exc}\n{abort_kind.detail}\n",
        )
    finally:
        try:
            adapter.stop()
        finally:
            adapter.cleanup()

    # Per-query latencies, kept as a raw artefact so `compare` can pair two runs
    # by query and run the paired test invariant I14 requires. Summaries cannot
    # be paired, and this is the only place the per-query values exist.
    bundle.write_raw_text(
        "latency-by-query.json",
        _as_json(
            {
                "queries_are_seeded_and_ordered": True,
                "points": [
                    {
                        "label": point.label,
                        "repetitions": [
                            {
                                "repetition": rep.repetition,
                                "latency_ms_by_query": {
                                    str(q): v for q, v in sorted(rep.latency_by_query.items())
                                },
                            }
                            for rep in point.repetitions
                        ],
                    }
                    for point in points
                ],
            }
        ),
    )

    telemetry = collectors.as_dict()
    bundle.write_raw_text("telemetry.json", _as_json(telemetry))

    # Phase 9 -- validation, over observed facts only.
    escapes = _escaped_pids(request.isolation)
    observations = RunObservations(
        observed_operations=sum(
            repetition.successes for point in points for repetition in point.repetitions
        ),
        expected_operations=_expected_operations(request, points),
        repetitions_declared=request.repetitions,
        repetitions_completed=min(
            (len(point.repetitions) for point in points if point.status == "measured"),
            default=0 if points else request.repetitions,
        ),
        timeouts=sum(r.timeouts for p in points for r in p.repetitions),
        errors=sum(r.errors for p in points for r in p.repetitions),
        sut_crashed=abort_kind is AbortKind.CRASHED,
        run_refused=abort_kind is AbortKind.REFUSED,
        budget_exceeded=abort_kind is AbortKind.BUDGET_EXCEEDED,
        oom_observed=oom,
        escaped_processes=escapes,
        cpu_limit_respected=_cpu_limit_respected(applied, escapes),
        memory_limit_respected=applied.memory_limit_applied,
        quality_reported=request.workload.quality_was_reported(points),
        quality_required=True,
        telemetry_complete=_telemetry_complete(telemetry),
        dirty_source_tree=_dirty_tree(environment),
    )
    validation = validate_run(observations, profile)
    bundle.write_artifact("validation", validation)
    bundle.write_artifact("result", _result_payload(bundle.run_id, points))

    statistics = summarise_points(
        [(point.label, point.parameters, point.metric_series()) for point in points]
    )

    # Phase 10 -- finalization. The manifest is written last.
    manifest = _manifest_payload(request, bundle, adapter, validation, environment)
    bundle.finalize(manifest)
    bundle.write_derived("statistics", statistics_payload(bundle.run_id, statistics))

    return RunOutcome(
        bundle=bundle,
        status=validation["status"],
        points=points,
        statistics=statistics,
        validation=validation,
        environment=environment,
        isolation=applied,
        telemetry=telemetry,
    )


# --------------------------------------------------------------------- helpers


def _as_json(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, default=str) + "\n"


def _build_collectors(request: RunRequest) -> CollectorSet:
    collectors = CollectorSet()
    pid = os.getpid()
    collectors.add(ProcessCollector(pid, enabled=request.collect_process_telemetry))
    collectors.add(PerfStatCollector(pid, enabled=request.collect_perf_telemetry))
    return collectors


def _expected_operations(request: RunRequest, points: list[PointResult]) -> int | None:
    measured = [point for point in points if point.status == "measured"]
    if not measured:
        return None
    return request.workload.expected_operations(len(measured), request.repetitions)


def _escaped_pids(plan: IsolationPlan) -> tuple[int, ...]:
    if plan.cpu_set is None:
        return ()
    return tuple(escape.pid for escape in find_escapes(os.getpid(), plan.cpu_set))


def _cpu_limit_respected(applied: AppliedIsolation, escapes: tuple[int, ...]) -> Any:
    if isinstance(applied.cpu_affinity_applied, Absent):
        return applied.cpu_affinity_applied
    return not escapes


def _telemetry_complete(telemetry: dict[str, Any]) -> bool:
    metrics = telemetry.get("metrics", {})
    enabled = telemetry.get("enabled", [])
    if not enabled:
        return True
    return any(
        not isinstance(value, dict)
        for key, value in metrics.items()
        if key.split(".", 1)[0] in enabled
    )


def _dirty_tree(environment: dict[str, Any]) -> Any:
    value = environment.get("source_control", {}).get("benchmark_dirty")
    if isinstance(value, dict) and "absent" in value:
        from theodb_bench.absent import AbsenceReason

        return Absent(AbsenceReason(value["absent"]), value.get("detail"))
    return bool(value)


def _manifest_payload(
    request: RunRequest,
    bundle: RunBundle,
    adapter: SystemAdapter,
    validation: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    source = environment.get("source_control", {})
    commit = source.get("benchmark_commit")
    dirty = source.get("benchmark_dirty")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": bundle.run_id,
        "status": validation["status"],
        "profile": request.profile.name.value,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "benchmark": {
            "id": request.benchmark_id,
            "version": request.benchmark_version,
            "profile": request.profile.name.value,
            "benchmark_commit": commit if isinstance(commit, str) else None,
            "benchmark_dirty": dirty if isinstance(dirty, bool) else None,
        },
        "system": {"id": adapter.system_id},
        "execution": {
            "warmup_operations": request.workload.warmup_operations,
            "repetitions": request.repetitions,
            "loop": "closed",
        },
        "environment_ref": "environment.json",
        "benchmark_ref": "benchmark.json",
        "system_ref": "system.json",
        "validation_ref": "validation.json",
    }
    if request.dataset_id is not None:
        manifest["dataset"] = {
            "id": request.dataset_id,
            "version": request.dataset_version,
            "sha256": request.dataset_sha256,
        }
    return manifest


def default_cpu_plan() -> IsolationPlan:
    """Every CPU this process may use. Declares the allocation without narrowing it."""
    return IsolationPlan(cpu_set=online_cpus())
