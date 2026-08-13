"""A finalized bundle is evidence; evidence that can be rewritten is not evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from theodb_bench.bundle import RUN_ID_PATTERN, RunBundle, build_run_id, slugify
from theodb_bench.errors import ImmutableBundleError, SchemaValidationError

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _bundle(tmp_path: Path) -> RunBundle:
    return RunBundle.create(
        tmp_path / "results",
        benchmark_id="vector/sift1m/hnsw",
        system_id="theodb",
        now=datetime(2026, 8, 12, 23, 10, 0, tzinfo=timezone.utc),
        entropy="deterministic",
    )


def _manifest_for(bundle: RunBundle) -> dict[str, Any]:
    manifest: dict[str, Any] = _fixture("manifest")
    manifest["run_id"] = bundle.run_id
    return manifest


# ------------------------------------------------------------------- run ids


def test_run_id_matches_the_manifest_schema_pattern() -> None:
    run_id = build_run_id("vector/sift1m/hnsw", "theodb")
    assert RUN_ID_PATTERN.match(run_id), run_id


def test_run_id_is_deterministic_for_the_same_inputs() -> None:
    now = datetime(2026, 8, 12, 23, 10, 0, tzinfo=timezone.utc)
    first = build_run_id("vector/sift1m/hnsw", "theodb", now=now, entropy="x")
    second = build_run_id("vector/sift1m/hnsw", "theodb", now=now, entropy="x")
    assert first == second


def test_run_id_encodes_benchmark_and_system() -> None:
    run_id = build_run_id("vector/sift1m/hnsw", "pgvector")
    assert "vector-sift1m-hnsw" in run_id
    assert "pgvector" in run_id


def test_slugify_strips_separators() -> None:
    assert slugify("vector/SIFT1M/hnsw") == "vector-sift1m-hnsw"


# -------------------------------------------------------------------- create


def test_create_makes_the_expected_layout(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    assert (bundle.root / "raw").is_dir()
    assert (bundle.root / "derived").is_dir()
    assert (bundle.root / "report").is_dir()
    assert not bundle.finalized


def test_create_refuses_to_reuse_a_directory(tmp_path: Path) -> None:
    _bundle(tmp_path)
    with pytest.raises(ImmutableBundleError, match="already exists"):
        _bundle(tmp_path)


def test_open_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ImmutableBundleError, match="no run bundle"):
        RunBundle.open(tmp_path / "nowhere")


# ----------------------------------------------------------------- artifacts


def test_artifacts_are_validated_before_being_written(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    broken = _fixture("system")
    broken["system"] = "Not A Valid Id"
    with pytest.raises(SchemaValidationError):
        bundle.write_artifact("system", broken)
    assert "system" not in bundle.artifacts()


def test_unknown_artifact_name_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ImmutableBundleError, match="not a bundle root artifact"):
        bundle.write_artifact("secrets", {})


def test_round_trip_through_disk(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    payload = _fixture("system")
    bundle.write_artifact("system", payload)
    assert bundle.read_artifact("system") == payload


# ------------------------------------------------------------------ raw data


def test_raw_paths_may_not_escape_the_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ImmutableBundleError, match="escapes the bundle"):
        bundle.raw_path("../../etc/passwd")


def test_raw_jsonl_appends_one_record_per_line(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.append_raw_jsonl("client.jsonl", {"op": 1, "latency_ms": 1.5})
    bundle.append_raw_jsonl("client.jsonl", {"op": 2, "latency_ms": 1.7})
    lines = (bundle.raw_dir / "client.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["op"] for line in lines] == [1, 2]


def test_raw_files_are_listed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.write_raw_text("system.log", "started\n")
    assert [p.name for p in bundle.raw_files()] == ["system.log"]


# ------------------------------------------------------------------ finalize


def test_finalize_writes_the_manifest_and_marks_the_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.finalize(_manifest_for(bundle))
    assert bundle.finalized
    assert bundle.read_artifact("manifest")["run_id"] == bundle.run_id


def test_finalize_rejects_a_manifest_for_another_run(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = _fixture("manifest")
    with pytest.raises(ImmutableBundleError, match="does not match bundle"):
        bundle.finalize(manifest)


def test_finalize_twice_is_refused(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.finalize(_manifest_for(bundle))
    with pytest.raises(ImmutableBundleError, match="finalized"):
        bundle.finalize(_manifest_for(bundle))


def test_finalized_bundle_refuses_new_root_artifacts(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.finalize(_manifest_for(bundle))
    with pytest.raises(ImmutableBundleError, match="would rewrite evidence"):
        bundle.write_artifact("system", _fixture("system"))


def test_finalized_bundle_refuses_new_raw_data(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.finalize(_manifest_for(bundle))
    with pytest.raises(ImmutableBundleError):
        bundle.write_raw_text("late.log", "should not be possible")


def test_finalized_raw_files_are_read_only_on_disk(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.write_raw_text("client.jsonl", '{"op":1}\n')
    bundle.finalize(_manifest_for(bundle))
    path = bundle.raw_dir / "client.jsonl"
    assert not path.stat().st_mode & 0o222


# --------------------------------------------------------------- re-analysis


def test_reanalysis_may_add_derived_output_after_finalization(tmp_path: Path) -> None:
    # Separating orchestration from reporting is only useful if re-analysis is
    # allowed to produce something.
    bundle = _bundle(tmp_path)
    bundle.finalize(_manifest_for(bundle))
    bundle.write_derived("statistics", _fixture("statistics"))
    assert bundle.read_artifact("statistics")["run_id"]


def test_reanalysis_may_not_overwrite_an_earlier_derivation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.write_derived("statistics", _fixture("statistics"))
    with pytest.raises(ImmutableBundleError, match="must not overwrite"):
        bundle.write_derived("statistics", _fixture("statistics"))


def test_explicit_overwrite_is_allowed_only_while_open(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.write_derived("statistics", _fixture("statistics"))
    bundle.write_derived("statistics", _fixture("statistics"), overwrite=True)
    bundle.finalize(_manifest_for(bundle))
    with pytest.raises(ImmutableBundleError):
        bundle.write_derived("statistics", _fixture("statistics"), overwrite=True)


def test_derived_artifact_name_is_checked(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ImmutableBundleError, match="not a derived artifact"):
        bundle.write_derived("manifest", {})


def test_artifacts_reports_what_the_bundle_actually_holds(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.write_artifact("system", _fixture("system"))
    bundle.write_derived("pareto", _fixture("pareto"))
    assert set(bundle.artifacts()) == {"system", "pareto"}
