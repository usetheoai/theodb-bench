"""Reports: the human half and the machine half of a result.

A report renders; it never measures and never re-decides anything. If a number
is not in the bundle, it does not appear here -- there is no path in this module
that computes a measurement.

What a report must not do is quietly present an invalid or non-publishable run
as evidence. Status and profile lead, before any number, because a reader who
skims the table has already been told what it is worth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from theodb_bench.absent import Absent
from theodb_bench.bundle import RunBundle
from theodb_bench.errors import ErrorContext, Phase, SchemaValidationError
from theodb_bench.profiles import get_profile
from theodb_bench.schemas import validate

SUMMARY_SCHEMA_VERSION: Final[int] = 1

_NOT_PUBLISHABLE = (
    "This result is **not publishable evidence**: the profile it ran under does "
    "not freeze methodology or datasets."
)
_INVALID = (
    "This run is **INVALID**. It failed protocol validation, which says nothing "
    "about whether the numbers were favourable -- invalidation is never based on "
    "the measured outcome."
)
_EXPLORATORY = (
    "This run is **EXPLORATORY**. Research runs may use non-frozen parameters, so "
    "the numbers below cannot back a published claim."
)


def _render(value: Any) -> str:
    """Render a value, making an absence visible rather than blank."""
    if isinstance(value, Absent):
        return f"_{value.reason.value}_"
    if isinstance(value, dict) and "absent" in value:
        return f"_{value['absent']}_"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if value is None:
        return "_none_"
    return str(value)


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()[:32]


def _headline(statistics: dict[str, Any]) -> dict[str, Any]:
    """The metrics a reader sees first, taken from the best measured point.

    "Best" means highest throughput among points that recorded one -- stated
    here rather than left implicit, since any choice of headline is a choice.
    """
    best: dict[str, Any] = {}
    best_throughput = float("-inf")
    for point in statistics.get("points", []):
        metrics = point.get("metrics", {})
        throughput = metrics.get("throughput_per_second", {}).get("median")
        if not isinstance(throughput, (int, float)):
            continue
        if throughput > best_throughput:
            best_throughput = throughput
            best = {
                "configuration": point["label"],
                "throughput_per_second": throughput,
                **{
                    name: metrics[name]["median"]
                    for name in ("recall", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
                    if name in metrics
                },
            }
    return best


def summary_payload(bundle: RunBundle) -> dict[str, Any]:
    """The machine half of the result: ``report/summary.json``."""
    manifest = bundle.read_artifact("manifest")
    environment = bundle.read_artifact("environment")
    try:
        statistics = bundle.read_artifact("statistics")
    except SchemaValidationError:
        statistics = {"points": []}

    profile_name = manifest.get("profile", "research")
    profile = get_profile(profile_name)
    limitations: list[str] = []
    if not profile.publishable:
        limitations.append(f"profile {profile_name!r} is not publishable")
    if manifest["status"] != "VALID":
        limitations.append(f"run status is {manifest['status']}")
    for point in statistics.get("points", []):
        stability = point.get("stability", {})
        if not stability.get("stable", True):
            limitations.append(f"{point['label']}: unstable ({stability.get('reason', '')})")

    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "profile": profile_name,
        "publishable": profile.publishable and manifest["status"] == "VALID",
        "benchmark": {
            "id": manifest["benchmark"]["id"],
            "version": manifest["benchmark"]["version"],
            "commit": manifest["benchmark"].get("benchmark_commit"),
        },
        "system": {"id": manifest["system"]["id"], "version": manifest["system"].get("version")},
        "environment_digest": _digest(environment),
        "headline": _headline(statistics),
        "limitations": limitations,
    }
    if manifest.get("dataset"):
        payload["dataset"] = manifest["dataset"]
    validate(
        "summary", payload, context=ErrorContext(phase=Phase.OFFLINE, run_id=payload["run_id"])
    )
    return payload


def render_markdown(bundle: RunBundle) -> str:
    """The human half of the result: ``report/report.md``."""
    manifest = bundle.read_artifact("manifest")
    environment = bundle.read_artifact("environment")
    validation = bundle.read_artifact("validation")
    try:
        statistics = bundle.read_artifact("statistics")
    except SchemaValidationError:
        statistics = {"points": []}

    profile_name = manifest.get("profile", "research")
    profile = get_profile(profile_name)
    status = manifest["status"]

    lines: list[str] = [
        f"# {manifest['benchmark']['id']} on {manifest['system']['id']}",
        "",
        f"**Status:** {status} · **Profile:** {profile_name} · **Run:** `{manifest['run_id']}`",
        "",
    ]

    if status == "INVALID":
        lines += [f"> {_INVALID}", ""]
    elif status == "EXPLORATORY":
        lines += [f"> {_EXPLORATORY}", ""]
    elif not profile.publishable:
        lines += [f"> {_NOT_PUBLISHABLE}", ""]

    lines += _results_section(statistics)
    lines += _validation_section(validation)
    lines += _environment_section(environment, manifest)
    return "\n".join(lines) + "\n"


def _median(metrics: dict[str, Any], name: str) -> str:
    """A point's median for one metric, or an explicit statement that it has none."""
    entry = metrics.get(name)
    return _render(entry["median"]) if entry else "_not measured_"


def _results_section(statistics: dict[str, Any]) -> list[str]:
    points = statistics.get("points", [])
    if not points:
        return ["## Results", "", "No configuration produced a measurement.", ""]

    columns = ["Configuration", "Throughput/s", "Recall", "p50 ms", "p95 ms", "p99 ms", "Stable"]
    lines = ["## Results", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for point in points:
        metrics = point.get("metrics", {})
        stability = point.get("stability", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    point["label"],
                    _median(metrics, "throughput_per_second"),
                    _median(metrics, "recall"),
                    _median(metrics, "latency_p50_ms"),
                    _median(metrics, "latency_p95_ms"),
                    _median(metrics, "latency_p99_ms"),
                    "yes" if stability.get("stable") else "**no**",
                ]
            )
            + " |"
        )
    lines.append("")

    unstable = [p for p in points if not p.get("stability", {}).get("stable", True)]
    if unstable:
        lines += [
            "Unstable points are reported, not removed. Their repetitions disagree "
            "by more than the declared threshold, so the median below is a weaker "
            "claim than it looks:",
            "",
        ]
        lines += [
            f"- `{p['label']}`: {p.get('stability', {}).get('reason', 'unknown')}" for p in unstable
        ]
        lines.append("")

    lines += ["### Repetitions", "", "Every repetition is retained:", ""]
    for point in points:
        for name, metric in sorted(point.get("metrics", {}).items()):
            values = ", ".join(_render(v) for v in metric.get("values", []))
            lines.append(f"- `{point['label']}` {name}: {values}")
    lines.append("")
    return lines


def _validation_section(validation: dict[str, Any]) -> list[str]:
    lines = ["## Validation", "", "| Check | Outcome | Required | Detail |", "|---|---|---|---|"]
    for check in validation.get("checks", []):
        lines.append(
            f"| {check['id']} | {check['outcome']} | "
            f"{'yes' if check.get('required') else 'no'} | {check.get('detail', '')} |"
        )
    lines.append("")
    if validation.get("invalidated_by"):
        lines += [
            "Invalidated by: " + ", ".join(f"`{c}`" for c in validation["invalidated_by"]),
            "",
            "Invalidation is based on protocol criteria, never on whether the numbers looked good.",
            "",
        ]
    return lines


def _environment_section(environment: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    cpu = environment.get("cpu", {})
    memory = environment.get("memory", {})
    software = environment.get("software", {})
    source = manifest.get("benchmark", {})
    return [
        "## Environment",
        "",
        f"- Host: {environment.get('hostname', 'unknown')}",
        f"- CPU: {_render(cpu.get('model'))} "
        f"({_render(cpu.get('logical_cpus'))} logical, "
        f"{_render(cpu.get('physical_cores'))} physical)",
        f"- SMT: {_render(cpu.get('smt_enabled'))} · "
        f"Governor: {_render(cpu.get('frequency_policy'))}",
        f"- Memory: {_render(memory.get('total_bytes'))} bytes",
        f"- Kernel: {_render(software.get('kernel'))} · "
        f"Runner: {_render(software.get('benchmark_runner'))}",
        f"- Benchmark commit: {_render(source.get('benchmark_commit'))} "
        f"(dirty: {_render(source.get('benchmark_dirty'))})",
        "",
        "Fields shown in italics were not available on this host and are recorded "
        "as absent rather than as zero.",
        "",
    ]


def write_report(bundle: RunBundle) -> tuple[Path, Path]:
    """Write both halves of the report into the bundle.

    Permitted after finalization: a report is derived from measurements, and
    writing one does not touch them.
    """
    bundle.report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = bundle.report_dir / "report.md"
    summary_path = bundle.report_dir / "summary.json"
    markdown_path.write_text(render_markdown(bundle), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload(bundle), indent=2) + "\n", encoding="utf-8")
    return markdown_path, summary_path


def render_comparison(bundles: list[RunBundle]) -> str:
    """A comparison table across systems, with each side's status visible.

    Systems appear with their run status and profile next to their numbers. A
    comparison that hides that one side was INVALID is not a comparison.
    """
    if not bundles:
        return "No runs to compare.\n"

    lines = [
        "# Comparison",
        "",
        "| System | Status | Profile | Configuration | Throughput/s | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for bundle in bundles:
        manifest = bundle.read_artifact("manifest")
        try:
            statistics = bundle.read_artifact("statistics")
        except SchemaValidationError:
            statistics = {"points": []}
        for point in statistics.get("points", []):
            metrics = point.get("metrics", {})
            throughput = metrics.get("throughput_per_second", {}).get("median")
            recall = metrics.get("recall", {}).get("median")
            lines.append(
                f"| {manifest['system']['id']} | {manifest['status']} | "
                f"{manifest.get('profile', '?')} | {point['label']} | "
                f"{_render(throughput)} | {_render(recall)} |"
            )
    lines += [
        "",
        "Each row carries the status and profile of the run it came from. A "
        "number from an INVALID run is shown so it can be seen, not so it can "
        "be used.",
        "",
    ]
    return "\n".join(lines) + "\n"
