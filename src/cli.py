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
from theodb_bench.doctor import render_report, run_doctor
from theodb_bench.environment import capture_environment
from theodb_bench.errors import BenchError
from theodb_bench.profiles import PROFILES, ProfileName, get_profile
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
