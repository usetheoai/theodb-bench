"""A report renders what was measured and never re-decides anything."""

from __future__ import annotations

import json
from pathlib import Path

from theodb_bench.adapters.fake import FakeAdapter, FakeConfig, Fault
from theodb_bench.bench.vector import VectorWorkload
from theodb_bench.profiles import get_profile
from theodb_bench.report import render_comparison, render_markdown, summary_payload, write_report
from theodb_bench.runner import RunOutcome, RunRequest, run_benchmark
from theodb_bench.schemas import validate


def _run(tmp_path: Path, **overrides: object) -> RunOutcome:
    base: dict[str, object] = {
        "benchmark_id": "vector/synthetic/smoke",
        "workload": VectorWorkload(corpus_size=128, dimension=8, query_count=16, k=4),
        "adapter_factory": FakeAdapter,
        "results_root": tmp_path / "results",
        "repetitions": 2,
    }
    base.update(overrides)
    return run_benchmark(RunRequest(**base))  # type: ignore[arg-type]


# -------------------------------------------------------------------- summary


def test_summary_validates_against_its_schema(tmp_path: Path) -> None:
    validate("summary", summary_payload(_run(tmp_path).bundle))


def test_a_smoke_run_is_never_marked_publishable(tmp_path: Path) -> None:
    payload = summary_payload(_run(tmp_path).bundle)
    assert payload["publishable"] is False
    assert any("not publishable" in note for note in payload["limitations"])


def test_an_invalid_run_is_not_publishable_whatever_the_profile(tmp_path: Path) -> None:
    outcome = _run(
        tmp_path,
        adapter_factory=lambda: FakeAdapter(FakeConfig(fault=Fault.CRASH, fail_after_queries=2)),
    )
    payload = summary_payload(outcome.bundle)
    assert payload["status"] == "INVALID"
    assert payload["publishable"] is False


def test_summary_carries_provenance(tmp_path: Path) -> None:
    payload = summary_payload(_run(tmp_path).bundle)
    assert payload["run_id"]
    assert payload["benchmark"]["id"] == "vector/synthetic/smoke"
    assert payload["environment_digest"].startswith("sha256:")


def test_environment_digest_is_stable_for_the_same_capture(tmp_path: Path) -> None:
    bundle = _run(tmp_path).bundle
    assert (
        summary_payload(bundle)["environment_digest"]
        == (summary_payload(bundle)["environment_digest"])
    )


def test_headline_names_the_configuration_it_came_from(tmp_path: Path) -> None:
    headline = summary_payload(_run(tmp_path).bundle)["headline"]
    assert "configuration" in headline
    assert "throughput_per_second" in headline


def test_unstable_points_are_listed_as_limitations(tmp_path: Path) -> None:
    outcome = _run(tmp_path, repetitions=3)
    payload = summary_payload(outcome.bundle)
    unstable = [p for p in outcome.statistics if not p.stability.stable]
    if unstable:
        assert any("unstable" in note for note in payload["limitations"])


# ------------------------------------------------------------------- markdown


def test_markdown_leads_with_status_and_profile(tmp_path: Path) -> None:
    text = render_markdown(_run(tmp_path).bundle)
    header, status_line = text.splitlines()[0], text.splitlines()[2]
    assert header.startswith("# vector/synthetic/smoke")
    assert "**Status:**" in status_line
    assert "**Profile:** smoke" in status_line


def test_markdown_warns_that_a_smoke_run_is_not_evidence(tmp_path: Path) -> None:
    assert "not publishable evidence" in render_markdown(_run(tmp_path).bundle)


def test_markdown_warns_loudly_on_an_invalid_run(tmp_path: Path) -> None:
    outcome = _run(
        tmp_path,
        adapter_factory=lambda: FakeAdapter(FakeConfig(fault=Fault.CRASH, fail_after_queries=2)),
    )
    text = render_markdown(outcome.bundle)
    assert "**INVALID**" in text
    assert "never based on the measured outcome" in text


def test_markdown_marks_an_exploratory_run(tmp_path: Path) -> None:
    outcome = _run(tmp_path, profile=get_profile("research"), repetitions=1)
    assert "**EXPLORATORY**" in render_markdown(outcome.bundle)


def test_markdown_shows_every_repetition(tmp_path: Path) -> None:
    outcome = _run(tmp_path, repetitions=3)
    text = render_markdown(outcome.bundle)
    assert "Every repetition is retained" in text
    for point in outcome.statistics:
        assert point.label in text


def test_markdown_includes_the_validation_table(tmp_path: Path) -> None:
    text = render_markdown(_run(tmp_path).bundle)
    assert "## Validation" in text
    assert "sut_alive" in text


def test_markdown_renders_absences_rather_than_blanks(tmp_path: Path) -> None:
    text = render_markdown(_run(tmp_path).bundle)
    assert "recorded as absent rather than as zero" in text
    # No table cell may be empty: an empty cell reads as zero to a skimming eye.
    for line in text.splitlines():
        if line.startswith("|") and "---" not in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            assert all(cells[:2]), line


def test_markdown_names_unstable_points_instead_of_dropping_them(tmp_path: Path) -> None:
    outcome = _run(tmp_path, repetitions=3)
    text = render_markdown(outcome.bundle)
    if any(not point.stability.stable for point in outcome.statistics):
        assert "reported, not removed" in text


# --------------------------------------------------------------------- write


def test_write_report_places_both_halves_in_the_bundle(tmp_path: Path) -> None:
    outcome = _run(tmp_path)
    markdown_path, summary_path = write_report(outcome.bundle)
    assert markdown_path.is_file()
    assert summary_path.is_file()
    validate("summary", json.loads(summary_path.read_text(encoding="utf-8")))


def test_writing_a_report_does_not_disturb_the_finalized_measurements(tmp_path: Path) -> None:
    outcome = _run(tmp_path)
    before = (outcome.bundle.root / "result.json").read_bytes()
    write_report(outcome.bundle)
    assert (outcome.bundle.root / "result.json").read_bytes() == before


# ----------------------------------------------------------------- comparison


def test_comparison_shows_each_side_status(tmp_path: Path) -> None:
    healthy = _run(tmp_path)
    broken = _run(
        tmp_path,
        adapter_factory=lambda: FakeAdapter(FakeConfig(fault=Fault.CRASH, fail_after_queries=2)),
    )
    text = render_comparison([healthy.bundle, broken.bundle])
    assert "INVALID" in text
    assert "so it can be seen, not so it can be used" in text


def test_comparison_of_nothing_says_so() -> None:
    assert "No runs to compare" in render_comparison([])
