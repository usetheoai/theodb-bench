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
import statistics
from pathlib import Path
from typing import Any, Final

from theodb_bench.absent import Absent
from theodb_bench.analysis.pareto import (
    MAXIMIZE,
    PARETO_SCHEMA_VERSION,
    Objective,
    Point,
    dominates,
    frontier,
)
from theodb_bench.bench.vector import PointResult
from theodb_bench.bundle import RunBundle
from theodb_bench.compare import match_by_recall, render_paired_verdict
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
    lines += _paired_verdicts(bundles)
    return "\n".join(lines) + "\n"


def _paired_verdicts(bundles: list[RunBundle]) -> list[str]:
    """The part of a comparison that is actually a comparison (I14).

    The table above is two summaries printed near each other, and a reader will
    infer a winner from it whether or not one exists. This section says whether
    the difference survives a paired randomisation test on the per-query
    latencies, in which direction, and by how much -- or says the runs cannot be
    paired.

    Exactly two runs are required. Three medians in a table invite a ranking; a
    paired test compares two things, and silently picking two of three to test
    would answer a question nobody asked.
    """
    if len(bundles) != 2:
        if len(bundles) > 2:
            return [
                "## Paired comparison",
                "",
                f"Not run: {len(bundles)} runs were given and a paired test compares "
                "two. Re-run `compare` with exactly two bundles.",
                "",
            ]
        return []

    samples = [_latency_by_query(b) for b in bundles]
    names = [b.read_artifact("manifest")["system"]["id"] for b in bundles]

    out = ["## Paired comparison", ""]

    shared = sorted(set(samples[0]) & set(samples[1]))
    if shared:
        # Same labels on both sides: a regression, one system measured twice.
        for label in shared:
            out.append(
                "- "
                + render_paired_verdict(
                    names[0],
                    samples[0][label],
                    names[1],
                    samples[1][label],
                    metric=f"latency_ms @ {label}",
                )
            )
        out += ["", _PAIRED_FOOTNOTE, ""]
        return out

    # Different engines. Their labels carry engine-specific knobs and can never
    # coincide -- `probes=20` on one side, `num_leaves_to_search=20` on the other --
    # so the frontiers are read on the quality axis both share.
    recalls = [_recall_by_label(b) for b in bundles]
    match = match_by_recall(recalls[0], recalls[1])
    if match is None:
        out += [
            "Not run: the two frontiers have no operating point at comparable "
            "recall (within 0.01). Pairing the nearest points regardless would "
            "compare a fast low-quality configuration against a slow high-quality "
            "one and report the first as a winner.",
            "",
        ]
        return out

    out += [
        f"Matched at recall {match.recall_a:.4f} ({names[0]}, `{match.label_a}`) "
        f"against {match.recall_b:.4f} ({names[1]}, `{match.label_b}`) — a gap of "
        f"{match.gap:.4f}.",
        "",
        "- "
        + render_paired_verdict(
            names[0],
            samples[0][match.label_a],
            names[1],
            samples[1][match.label_b],
            metric="latency_ms at matched recall",
        ),
        "",
        _PAIRED_FOOTNOTE,
        "",
    ]
    return out


_PAIRED_FOOTNOTE = (
    "Lower latency wins. `p` is a paired randomisation test with Monte-Carlo "
    "correction; the interval is a paired percentile bootstrap. An "
    "`indistinguishable` verdict is a result, not a missing one.\n\n"
    "**What the pairing does not control.** Pairing removes the variance of query "
    "difficulty: both systems answered the same query. It does not remove drift in "
    "the machine, because the two runs happened at different times. Measured on "
    "2026-08-17, the same configuration re-run on the same host varied by 24% and "
    "46% in median throughput, so a busier machine during one side of the pair is "
    "attributed to the engine with the same confidence a real difference would be. "
    "The interval does not protect against this: it measures dispersion across "
    "queries, not across runs. Interleaving the two systems query by query would "
    "control it, and this harness does not yet do that."
)


def _recall_by_label(bundle: RunBundle) -> dict[str, float]:
    """Median recall per configuration, for reading a frontier at matched quality."""
    try:
        statistics = bundle.read_artifact("statistics")
    except (SchemaValidationError, KeyError):
        return {}
    out: dict[str, float] = {}
    for point in statistics.get("points", []):
        recall = point.get("metrics", {}).get("recall", {}).get("median")
        if recall is not None:
            out[point["label"]] = float(recall)
    return out


def _latency_by_query(bundle: RunBundle) -> dict[str, dict[int, float]]:
    """Per-query latencies from a bundle, per configuration label.

    Repetitions are averaged per query before pairing: the paired unit is the
    query, and a query measured three times on each side is still one paired
    observation. Treating each repetition as independent would inflate `n`
    threefold and make any difference look significant.
    """
    try:
        raw = json.loads((bundle.raw_dir / "latency-by-query.json").read_text())
    except (OSError, ValueError):
        return {}

    out: dict[str, dict[int, float]] = {}
    for point in raw.get("points", []):
        totals: dict[int, list[float]] = {}
        for rep in point.get("repetitions", []):
            for qid, value in rep.get("latency_ms_by_query", {}).items():
                totals.setdefault(int(qid), []).append(float(value))
        if totals:
            out[point["label"]] = {q: sum(v) / len(v) for q, v in totals.items()}
    return out


def pareto_payload(points: list[PointResult]) -> dict[str, Any] | None:
    """The quality/throughput frontier of a swept run, or None when there is none.

    The project's own rule is that a headline throughput comparison needs a stated
    target quality with its interpolation method, *or* the complete frontier.
    `analysis/pareto.py` computed frontiers and had no caller, so no run emitted
    one and every comparison fell to the first branch by default.

    Returns None below two measured configurations. A frontier of one point is a
    point, and publishing it as a curve would dress a single measurement as a
    trade-off.
    """
    measured = [
        point
        for point in points
        if point.status == "measured" and point.repetitions and _median_recall(point) is not None
    ]
    if len(measured) < 2:
        return None

    objectives = [
        Objective(metric="throughput_per_second", direction=MAXIMIZE),
        Objective(metric="recall", direction=MAXIMIZE),
    ]
    candidates = [
        Point(
            label=point.label,
            values={
                "throughput_per_second": _median_throughput(point) or 0.0,
                "recall": _median_recall(point) or 0.0,
            },
        )
        for point in measured
    ]
    best = frontier(candidates, objectives)
    on_frontier = {point.label for point in best}
    return {
        "schema_version": PARETO_SCHEMA_VERSION,
        "objectives": [objective.as_dict() for objective in objectives],
        "points": [
            {
                "label": point.label,
                "values": point.values,
                # Who beat it, not merely that something did: an operator fixing a
                # dominated configuration needs to know which one to compare with.
                "dominated_by": [
                    other.label for other in candidates if dominates(other, point, objectives)
                ],
            }
            for point in candidates
        ],
        "frontier": [point.label for point in sorted(best, key=lambda p: p.values["recall"])],
    }


def _median_recall(point: PointResult) -> float | None:
    values = [r.recall for r in point.repetitions if r.recall is not None]
    return statistics.median(values) if values else None


def _median_throughput(point: PointResult) -> float | None:
    values = [r.throughput for r in point.repetitions if r.throughput is not None]
    return statistics.median(values) if values else None
