"""Command-line entry point.

Argparse rather than a CLI framework: subcommand dispatch is a solved problem
in the standard library, and a benchmark runner should not carry a dependency
it does not need (TRD section 37 -- the runner must not become the thing that
distorts the measurement).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from theodb_bench import __version__
from theodb_bench.bundle import RunBundle
from theodb_bench.datasets import (
    DatasetRegistry,
    default_registry,
    fetch_dataset,
    verify_dataset,
)
from theodb_bench.doctor import render_report, run_doctor
from theodb_bench.environment import capture_environment
from theodb_bench.errors import BenchError, ConfigError, ErrorContext, Phase
from theodb_bench.profiles import PROFILES, ProfileName, get_profile
from theodb_bench.registry import ADAPTERS, BENCHMARKS, get_adapter, get_benchmark
from theodb_bench.report import render_comparison, write_report
from theodb_bench.runner import RunRequest, run_benchmark
from theodb_bench.schemas import SCHEMA_NAMES, read_validated

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_PREFLIGHT_BLOCKED: Final[int] = 2

DEFAULT_DATASET_ROOT: Final[Path] = Path(".datasets")
DEFAULT_RESULTS_ROOT: Final[Path] = Path("results")


def _emit(payload: object, *, as_json: bool, human: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(human)


def cmd_doctor(args: argparse.Namespace) -> int:
    profile = get_profile(args.profile)
    report = run_doctor(profile, dataset_root=args.dataset_root)
    _emit(report.as_dict(), as_json=args.json, human=render_report(report))
    return EXIT_OK if report.may_run else EXIT_PREFLIGHT_BLOCKED


def cmd_env(args: argparse.Namespace) -> int:
    environment = capture_environment()
    if args.json:
        print(json.dumps(environment, indent=2))
        return EXIT_OK
    cpu = environment["cpu"]
    memory = environment["memory"]
    software = environment["software"]

    def show(value: object) -> str:
        if isinstance(value, dict) and "absent" in value:
            detail = value.get("detail")
            return f"<{value['absent']}{': ' + str(detail) if detail else ''}>"
        return str(value)

    print(f"host      {environment['hostname']}")
    print(f"os        {show(software['os'])}")
    print(f"kernel    {show(software['kernel'])}")
    print(f"cpu       {show(cpu['model'])}")
    print(f"topology  {show(cpu['logical_cpus'])} logical / {show(cpu['physical_cores'])} physical")
    print(f"memory    {show(memory['total_bytes'])} bytes")
    print(f"runner    {software['benchmark_runner']}")
    return EXIT_OK


def cmd_profiles(args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps(
                {
                    name.value: {
                        "min_repetitions": p.min_repetitions,
                        "telemetry": p.telemetry.value,
                        "isolation_required": p.isolation_required,
                        "preflight_required": p.preflight_required,
                        "publishable": p.publishable,
                        "regression_gate": p.regression_gate,
                        "frozen_methodology": p.frozen_methodology,
                        "description": p.description,
                    }
                    for name, p in PROFILES.items()
                },
                indent=2,
            )
        )
        return EXIT_OK
    width = max(len(name.value) for name in ProfileName)
    for name, profile in PROFILES.items():
        flags = "publishable" if profile.publishable else "not publishable"
        print(f"{name.value.ljust(width)}  {flags:<15}  {profile.description}")
    return EXIT_OK


def cmd_schema_list(args: argparse.Namespace) -> int:
    for name in SCHEMA_NAMES:
        print(name)
    return EXIT_OK


def cmd_schema_validate(args: argparse.Namespace) -> int:
    read_validated(args.schema, args.path)
    print(f"{args.path}: valid against {args.schema} schema")
    return EXIT_OK


def cmd_list(args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps(
                {
                    "benchmarks": {b.id: b.description for b in BENCHMARKS.values()},
                    "systems": {
                        a.name: {
                            "description": a.description,
                            "available": a.available,
                            "missing": a.unmet_requirements(),
                        }
                        for a in ADAPTERS.values()
                    },
                },
                indent=2,
            )
        )
        return EXIT_OK
    print("Benchmarks:")
    for entry in BENCHMARKS.values():
        print(f"  {entry.id}")
        print(f"      {entry.description}")
    print()
    print("Systems:")
    for adapter in ADAPTERS.values():
        # An adapter whose driver is missing stays visible with the reason:
        # "not listed" and "not installed" lead to different actions.
        state = "" if adapter.available else f"  [needs {', '.join(adapter.unmet_requirements())}]"
        print(f"  {adapter.name}{state}")
        print(f"      {adapter.description}")
    return EXIT_OK


def cmd_describe(args: argparse.Namespace) -> int:
    entry = get_benchmark(args.benchmark)
    workload = entry.workload
    workload_payload: dict[str, object] = {
        "corpus_size": workload.corpus_size,
        "dimension": workload.dimension,
        "query_count": workload.query_count,
        "k": workload.k,
        "metric": workload.metric,
        "seed": workload.seed,
        "warmup_queries": workload.warmup_queries,
        "indexes": [index.label() for index in workload.indexes],
        "search_sweep": {name: list(v) for name, v in workload.search_sweep.items()},
    }
    payload: dict[str, object] = {
        "id": entry.id,
        "description": entry.description,
        "default_repetitions": entry.default_repetitions,
        "workload": workload_payload,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(f"{entry.id}\n  {entry.description}\n")
    for key, value in workload_payload.items():
        print(f"  {key:16s} {value}")
    return EXIT_OK


def _dataset_registry(args: argparse.Namespace) -> DatasetRegistry:
    return DatasetRegistry(args.manifest_dir) if args.manifest_dir else default_registry()


def cmd_dataset_list(args: argparse.Namespace) -> int:
    registry = _dataset_registry(args)
    identifiers = registry.ids()
    if not identifiers:
        print(f"no dataset manifests in {registry.manifest_dir}")
        return EXIT_OK
    for manifest in registry.all():
        print(
            f"{manifest.id}  version={manifest.version}  files={len(manifest.files)}  "
            f"licence={manifest.license_name}"
        )
    return EXIT_OK


def cmd_dataset_verify(args: argparse.Namespace) -> int:
    manifest = _dataset_registry(args).load(args.dataset)
    verification = verify_dataset(manifest, args.dataset_root)
    if args.json:
        print(json.dumps(verification.as_dict(), indent=2))
    else:
        for entry in verification.files:
            state = "ok" if entry.ok else ("missing" if not entry.present else "MISMATCH")
            print(f"{state:9s} {entry.path}")
        print("verified" if verification.ok else "NOT verified")
    return EXIT_OK if verification.ok else EXIT_ERROR


def cmd_dataset_fetch(args: argparse.Namespace) -> int:
    manifest = _dataset_registry(args).load(args.dataset)
    verification = fetch_dataset(manifest, args.dataset_root, force=args.force)
    print(
        f"{manifest.id}: {len(verification.files)} file(s) verified in "
        f"{manifest.directory(args.dataset_root)}"
    )
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    entry = get_benchmark(args.benchmark)
    adapter = get_adapter(args.system)
    profile = get_profile(args.profile)
    repetitions = args.repetitions if args.repetitions is not None else entry.default_repetitions

    outcome = run_benchmark(
        RunRequest(
            benchmark_id=entry.id,
            workload=entry.workload,
            adapter_factory=adapter.build,
            profile=profile,
            repetitions=repetitions,
            results_root=args.output,
            collect_perf_telemetry=args.perf,
        )
    )
    write_report(outcome.bundle)

    print(f"run       {outcome.run_id}")
    print(f"status    {outcome.status}")
    print(f"bundle    {outcome.bundle.root}")
    for point in outcome.statistics:
        throughput = point.metrics.get("throughput_per_second")
        recall = point.metrics.get("recall")
        rendered_throughput = f"{throughput.median:,.1f}" if throughput else "not measured"
        rendered_recall = f"{recall.median:.4f}" if recall else "not measured"
        marker = "" if point.stability.stable else "  (unstable)"
        print(
            f"  {point.label:32s} qps={rendered_throughput:>12s}  recall={rendered_recall}{marker}"
        )
    if outcome.status != "VALID":
        reasons = ", ".join(outcome.validation["invalidated_by"]) or "see validation.json"
        print(f"\nRun is {outcome.status}: {reasons}")
    return EXIT_OK if outcome.status != "INVALID" else EXIT_ERROR


def cmd_report(args: argparse.Namespace) -> int:
    bundle = RunBundle.open(args.run_dir)
    markdown_path, summary_path = write_report(bundle)
    print(f"{markdown_path}\n{summary_path}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    bundles = [RunBundle.open(path) for path in args.run_dirs]
    print(render_comparison(bundles))
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = RunBundle.open(args.run_dir)
    present = bundle.artifacts()
    for name, path in sorted(present.items()):
        bundle.read_artifact(name)
        print(f"ok  {name:12s} {path.name}")
    manifest = bundle.read_artifact("manifest")
    print(f"\nstatus {manifest['status']}  run {manifest['run_id']}")
    if not bundle.finalized:
        raise ConfigError(
            f"bundle {bundle.run_id} was never finalized; its measurement did not complete",
            context=ErrorContext(phase=Phase.OFFLINE, run_id=bundle.run_id),
        )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="theodb-bench",
        description="Open, reproducible performance benchmarking for TheoDB.",
    )
    parser.add_argument("--version", action="version", version=f"theodb-bench {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="check whether this host may run a benchmark under a profile"
    )
    doctor.add_argument(
        "--profile",
        default=ProfileName.SMOKE.value,
        choices=[p.value for p in ProfileName],
        help="profile whose mandatory checks apply (default: smoke)",
    )
    doctor.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"where fetched datasets live (default: {DEFAULT_DATASET_ROOT})",
    )
    doctor.add_argument("--json", action="store_true", help="emit machine-readable output")
    doctor.set_defaults(func=cmd_doctor)

    env = subparsers.add_parser("env", help="capture and print the environment record")
    env.add_argument("--json", action="store_true", help="emit the full environment.json payload")
    env.set_defaults(func=cmd_env)

    profiles = subparsers.add_parser("profiles", help="list benchmark profiles and their rules")
    profiles.add_argument("--json", action="store_true", help="emit machine-readable output")
    profiles.set_defaults(func=cmd_profiles)

    schema = subparsers.add_parser("schema", help="inspect and validate artifact schemas")
    schema_sub = schema.add_subparsers(dest="schema_command", required=True)
    schema_list = schema_sub.add_parser("list", help="list known schemas")
    schema_list.set_defaults(func=cmd_schema_list)
    schema_validate = schema_sub.add_parser("validate", help="validate a JSON artifact")
    schema_validate.add_argument("schema", choices=list(SCHEMA_NAMES))
    schema_validate.add_argument("path", type=Path)
    schema_validate.set_defaults(func=cmd_schema_validate)

    listing = subparsers.add_parser("list", help="list benchmarks and systems")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    describe = subparsers.add_parser("describe", help="show a benchmark definition")
    describe.add_argument("benchmark", choices=sorted(BENCHMARKS))
    describe.add_argument("--json", action="store_true")
    describe.set_defaults(func=cmd_describe)

    dataset = subparsers.add_parser("dataset", help="manage benchmark datasets")
    dataset.add_argument("--manifest-dir", type=Path, default=None)
    dataset.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    dataset_list = dataset_sub.add_parser("list", help="list known datasets")
    dataset_list.set_defaults(func=cmd_dataset_list)

    dataset_verify = dataset_sub.add_parser("verify", help="verify a dataset against its manifest")
    dataset_verify.add_argument("dataset")
    dataset_verify.add_argument("--json", action="store_true")
    dataset_verify.set_defaults(func=cmd_dataset_verify)

    dataset_fetch = dataset_sub.add_parser("fetch", help="download and verify a dataset")
    dataset_fetch.add_argument("dataset")
    dataset_fetch.add_argument("--force", action="store_true", help="re-download mismatched files")
    dataset_fetch.set_defaults(func=cmd_dataset_fetch)

    run = subparsers.add_parser("run", help="execute a benchmark against a system")
    run.add_argument("benchmark", choices=sorted(BENCHMARKS))
    run.add_argument("--system", default="fake", choices=sorted(ADAPTERS))
    run.add_argument(
        "--profile", default=ProfileName.SMOKE.value, choices=[p.value for p in ProfileName]
    )
    run.add_argument("--repetitions", type=int, default=None)
    run.add_argument("--output", type=Path, default=DEFAULT_RESULTS_ROOT)
    run.add_argument("--perf", action="store_true", help="enable hardware counter collection")
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="render the report for an existing run")
    report.add_argument("run_dir", type=Path)
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser("compare", help="compare existing runs")
    compare.add_argument("run_dirs", type=Path, nargs="+")
    compare.set_defaults(func=cmd_compare)

    validate_cmd = subparsers.add_parser("validate", help="re-validate an existing run bundle")
    validate_cmd.add_argument("run_dir", type=Path)
    validate_cmd.set_defaults(func=cmd_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except BenchError as exc:
        # Typed errors carry their own context; print it and exit non-zero
        # rather than letting a traceback stand in for a diagnosis.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
