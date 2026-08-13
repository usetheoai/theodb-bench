"""End-to-end: a real run against the fake system, producing a real bundle.

Nothing is mocked here except the database itself. The phases execute in order,
artifacts are written and validated against their schemas, and the bundle is
frozen at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from theodb_bench.adapters.base import IndexSpec
from theodb_bench.adapters.fake import FakeAdapter, FakeConfig, Fault
from theodb_bench.bench.vector import VectorWorkload, build_label
from theodb_bench.bundle import RunBundle
from theodb_bench.errors import ImmutableBundleError, PreflightError
from theodb_bench.profiles import get_profile
from theodb_bench.runner import RunRequest, run_benchmark
from theodb_bench.schemas import validate


def _workload(**overrides: object) -> VectorWorkload:
    base: dict[str, object] = {
        "corpus_size": 256,
        "dimension": 8,
        "query_count": 24,
        "k": 5,
        "warmup_queries": 4,
    }
    base.update(overrides)
    return VectorWorkload(**base)  # type: ignore[arg-type]


def _request(tmp_path: Path, **overrides: object) -> RunRequest:
    base: dict[str, object] = {
        "benchmark_id": "vector/synthetic/smoke",
        "workload": _workload(),
        "adapter_factory": FakeAdapter,
        "results_root": tmp_path / "results",
        "repetitions": 2,
    }
    base.update(overrides)
    return RunRequest(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------- happy path


def test_a_clean_run_produces_a_valid_finalized_bundle(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    assert outcome.status == "VALID"
    assert outcome.bundle.finalized


def test_the_bundle_has_the_canonical_layout(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    root = outcome.bundle.root
    for name in ("manifest", "environment", "benchmark", "system", "validation", "result"):
        assert (root / f"{name}.json").is_file(), name
    assert (root / "derived" / "statistics.json").is_file()
    assert (root / "raw").is_dir()
    assert outcome.bundle.raw_files()


@pytest.mark.parametrize(
    "artifact", ["manifest", "environment", "benchmark", "system", "validation", "result"]
)
def test_every_artifact_validates_against_its_schema(tmp_path: Path, artifact: str) -> None:
    outcome = run_benchmark(_request(tmp_path))
    validate(artifact, outcome.bundle.read_artifact(artifact))


def test_statistics_validate_and_keep_every_repetition(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path, repetitions=3))
    payload = outcome.bundle.read_artifact("statistics")
    validate("statistics", payload)
    for point in payload["points"]:
        for metric in point["metrics"].values():
            assert metric["repetitions"] == 3
            assert len(metric["values"]) == 3


def test_exact_search_against_the_fake_gives_perfect_recall(tmp_path: Path) -> None:
    # The fake answers by brute force, so anything below 1.0 would mean the
    # recall computation itself is wrong.
    outcome = run_benchmark(_request(tmp_path))
    recalls = [
        repetition.recall
        for point in outcome.points
        for repetition in point.repetitions
        if repetition.recall is not None
    ]
    assert recalls
    assert all(value == pytest.approx(1.0) for value in recalls)


def test_the_run_id_encodes_benchmark_and_system(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    assert "vector-synthetic-smoke" in outcome.run_id
    assert "fake" in outcome.run_id


def test_warmup_queries_are_not_counted_as_measured_operations(tmp_path: Path) -> None:
    # Four warm-up queries per configuration must not appear in the totals.
    outcome = run_benchmark(_request(tmp_path, repetitions=1))
    measured = sum(r.successes for p in outcome.points for r in p.repetitions)
    assert measured == 24 * len([p for p in outcome.points if p.status == "measured"])


# ---------------------------------------------------------------- immutability


def test_the_finalized_bundle_refuses_further_writes(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    with pytest.raises(ImmutableBundleError):
        outcome.bundle.write_raw_text("late.log", "no")


def test_reanalysis_can_reopen_and_add_a_derivation(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    reopened = RunBundle.open(outcome.bundle.root)
    assert reopened.finalized
    assert reopened.read_artifact("manifest")["run_id"] == outcome.run_id
    reopened.write_derived(
        "pareto",
        {
            "schema_version": 1,
            "objectives": [{"metric": "recall", "direction": "maximize"}],
            "points": [{"label": "x", "values": {"recall": 1.0}}],
            "frontier": ["x"],
        },
    )


# --------------------------------------------------------------------- faults


def test_a_crash_mid_run_invalidates_the_bundle(tmp_path: Path) -> None:
    outcome = run_benchmark(
        _request(
            tmp_path,
            adapter_factory=lambda: FakeAdapter(
                FakeConfig(fault=Fault.CRASH, fail_after_queries=5)
            ),
        )
    )
    assert outcome.status == "INVALID"
    assert "sut_alive" in outcome.validation["invalidated_by"]


def test_an_oom_is_recorded_as_an_oom(tmp_path: Path) -> None:
    outcome = run_benchmark(
        _request(
            tmp_path,
            adapter_factory=lambda: FakeAdapter(FakeConfig(fault=Fault.OOM, fail_after_queries=2)),
        )
    )
    assert outcome.status == "INVALID"
    assert "no_oom" in outcome.validation["invalidated_by"]


def test_a_system_that_never_becomes_ready_invalidates_the_run(tmp_path: Path) -> None:
    outcome = run_benchmark(
        _request(tmp_path, adapter_factory=lambda: FakeAdapter(FakeConfig(fault=Fault.NOT_READY)))
    )
    assert outcome.status == "INVALID"


def test_a_quality_regression_shows_up_as_lower_recall_not_as_a_failure(tmp_path: Path) -> None:
    # The point of measuring quality separately: the run is technically valid
    # and the answers are worse.
    outcome = run_benchmark(
        _request(
            tmp_path,
            adapter_factory=lambda: FakeAdapter(FakeConfig(fault=Fault.QUALITY_REGRESSION)),
        )
    )
    recalls = [r.recall for p in outcome.points for r in p.repetitions if r.recall is not None]
    assert recalls
    assert max(recalls) < 1.0


def test_an_unsupported_index_is_recorded_as_unsupported_not_measured(tmp_path: Path) -> None:
    outcome = run_benchmark(
        _request(
            tmp_path,
            workload=_workload(indexes=(IndexSpec(kind="ivfflat", parameters={"lists": 4}),)),
        )
    )
    statuses = {point.status for point in outcome.points}
    assert statuses == {"unsupported"}
    result = outcome.bundle.read_artifact("result")
    assert result["points"][0]["status"] == "unsupported"
    assert "does not support" in result["points"][0]["status_detail"]


# -------------------------------------------------------------------- profiles


def test_release_requires_more_repetitions_than_requested(tmp_path: Path) -> None:
    from theodb_bench.errors import ConfigError

    with pytest.raises(ConfigError, match="at least 5"):
        run_benchmark(_request(tmp_path, profile=get_profile("release"), repetitions=2))


def test_a_profile_requiring_preflight_stops_on_a_blocked_host(tmp_path: Path) -> None:
    # release is blocked on any host with frequency scaling or swap enabled,
    # which is the common case for a developer machine.
    from theodb_bench.doctor import run_doctor

    report = run_doctor(get_profile("release"))
    if report.may_run:
        pytest.skip("this host satisfies every mandatory release check")
    with pytest.raises(PreflightError, match="blocking checks"):
        run_benchmark(_request(tmp_path, profile=get_profile("release"), repetitions=5))


def test_a_research_run_is_marked_exploratory(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path, profile=get_profile("research"), repetitions=1))
    assert outcome.status == "EXPLORATORY"


# ------------------------------------------------------------------ telemetry


def test_telemetry_records_its_own_overhead(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    assert outcome.telemetry["overhead_seconds"] >= 0
    assert "process" in outcome.telemetry["enabled"]


def test_disabled_telemetry_is_recorded_as_not_collected(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path, collect_process_telemetry=False))
    raw = json.loads((outcome.bundle.raw_dir / "telemetry.json").read_text(encoding="utf-8"))
    process_metrics = [v for k, v in raw["metrics"].items() if k.startswith("process.")]
    assert process_metrics
    assert all(v.get("absent") == "not_collected" for v in process_metrics)


# --------------------------------------------------------------------- labels


def test_a_query_cap_appears_in_the_label(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path, workload=_workload(query_cap=8)))
    assert all("[q=8]" in point.label for point in outcome.points)
    assert all(r.successes == 8 for p in outcome.points for r in p.repetitions)


def test_build_label_is_stable_and_readable() -> None:
    label = build_label(IndexSpec(kind="hnsw", parameters={"m": 16}), {"ef_search": 64}, None)
    assert label == "hnsw m=16 ef_search=64"
