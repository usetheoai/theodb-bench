"""Command-line entry point.

Argparse rather than a CLI framework: subcommand dispatch is a solved problem
in the standard library, and a benchmark runner should not carry a dependency
it does not need (TRD section 37 -- the runner must not become the thing that
distorts the measurement).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from theodb_bench import __version__
from theodb_bench.adapters.base import IndexSpec
from theodb_bench.bench.vector import VectorBenchmark, VectorWorkload, generate_corpus
from theodb_bench.bundle import RunBundle
from theodb_bench.compare import render_paired_verdict
from theodb_bench.datasets import (
    DatasetManifest,
    DatasetRegistry,
    default_registry,
    fetch_dataset,
    require_verified,
    verify_dataset,
)
from theodb_bench.doctor import render_report, run_doctor
from theodb_bench.environment import capture_environment
from theodb_bench.errors import BenchError, ConfigError, ErrorContext, Phase
from theodb_bench.formats import AnnDataset, read_ann_hdf5
from theodb_bench.interleaved import interleave
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
    # The workload describes itself. Enumerating its fields here would put one
    # family's vocabulary -- `k`, `dimension`, `search_sweep` -- in front of every
    # family, and `describe` would start lying about an analytical suite.
    workload_payload: dict[str, object] = {
        **workload.benchmark_payload(),
        "declared": {
            key: _describable(value)
            for key, value in dataclasses.asdict(workload).items()  # type: ignore[call-overload]
        },
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


def _load_ann_dataset(
    manifest: DatasetManifest, root: Path, workload: object
) -> tuple[AnnDataset, str]:
    """Verify a dataset, then read the vectors the run will actually measure.

    Verification happens first and unconditionally. Reading unverified bytes
    would produce a number about a dataset nobody can identify, and the
    manifest would still name one.
    """
    verification = require_verified(manifest, root)
    entry = manifest.file_by_role("train") or (manifest.files[0] if manifest.files else None)
    if entry is None:
        raise ConfigError(
            f"dataset {manifest.id} declares no files",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    path = manifest.resolve(root, entry)
    if path.suffix.lower() not in {".hdf5", ".h5"}:
        raise ConfigError(
            f"dataset {manifest.id}: only ANN-Benchmarks HDF5 files can be loaded today; "
            f"{path.name} is not one",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    dataset = read_ann_hdf5(path)
    digest = next((f.expected_sha256 for f in verification.files if f.path == entry.path), "")
    return dataset, digest


def cmd_run(args: argparse.Namespace) -> int:
    entry = get_benchmark(args.benchmark)
    adapter = get_adapter(args.system)
    profile = get_profile(args.profile)
    repetitions = args.repetitions if args.repetitions is not None else entry.default_repetitions

    workload = entry.workload
    corpus = queries = None
    dataset_id = dataset_version = dataset_sha256 = None

    if args.dataset:
        if not isinstance(workload, VectorWorkload):
            raise ConfigError(
                f"`{entry.id}` generates its own data from its seed, and "
                f"`--dataset {args.dataset}` supplies an ANN corpus. Accepting it "
                f"would record a dataset identity the run never measured.",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        manifest = _dataset_registry(args).load(args.dataset)
        loaded, dataset_sha256 = _load_ann_dataset(manifest, args.dataset_root, workload)
        reduced = loaded.subsample(
            min(workload.corpus_size, loaded.corpus_size),
            min(workload.query_count, loaded.query_count),
        )
        # The workload adopts the data's own shape rather than the other way
        # round: reshaping real vectors to fit a declared dimension would
        # measure something the dataset does not contain.
        workload = replace(
            workload,
            dimension=reduced.dimension,
            corpus_size=reduced.corpus_size,
            query_count=reduced.query_count,
        )
        corpus, queries = reduced.train, reduced.test
        dataset_id, dataset_version = manifest.id, manifest.version
        print(
            f"dataset   {manifest.id} v{manifest.version}: verified, "
            f"{reduced.corpus_size} vectors x {reduced.dimension} dims, "
            f"{reduced.query_count} queries"
        )

    outcome = run_benchmark(
        RunRequest(
            benchmark_id=entry.id,
            workload=workload,
            adapter_factory=adapter.build,
            profile=profile,
            repetitions=repetitions,
            results_root=args.output,
            collect_perf_telemetry=args.perf,
            corpus=corpus,
            queries=queries,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
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


def _describable(value: object) -> object:
    """Coerce a declared workload field into something JSON can carry."""
    if isinstance(value, tuple):
        return [_describable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _describable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def cmd_head2head(args: argparse.Namespace) -> int:
    """Two systems, same queries, back to back, with the order alternating.

    Sequential runs leave machine drift in the comparison: measured on
    2026-08-17, the same configuration re-run on the same host varied by 24% and
    46% in median throughput, and a paired test attributes that to the engine with
    the same confidence a real difference would get. Interleaving sends query *i*
    to both systems before moving on, so anything moving on the scale of minutes
    moves both sides together.

    Each system gets its own benchmark, because two engines need different index
    configurations to reach the same quality and their knobs are not the same
    knobs -- `pq_subspaces` is meaningless to AlloyDB and `num_leaves` is
    meaningless to us. What must match is the experiment: corpus, queries, k and
    metric. That is checked rather than assumed.
    """
    entry_a = get_benchmark(args.benchmark_a)
    entry_b = get_benchmark(args.benchmark_b)
    for entry in (entry_a, entry_b):
        if not isinstance(entry.workload, VectorWorkload):
            raise ConfigError(
                f"head2head interleaves k-NN probes and `{entry.id}` is not a vector "
                f"benchmark. Interleaving an analytical suite would need a different "
                f"unit of work on both sides, and pretending otherwise would compare "
                f"two things that are not the same measurement.",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
    _require_comparable(entry_a.workload, entry_b.workload)

    workload = entry_a.workload
    if not isinstance(workload, VectorWorkload):  # refused above; narrows the type
        raise ConfigError(
            f"`{entry_a.id}` is not a vector benchmark",
            context=ErrorContext(phase=Phase.PREFLIGHT),
        )
    corpus, queries = generate_corpus(workload)
    if args.dataset:
        manifest = _dataset_registry(args).load(args.dataset)
        loaded, _ = _load_ann_dataset(manifest, args.dataset_root, workload)
        reduced = loaded.subsample(
            min(workload.corpus_size, loaded.corpus_size),
            min(workload.query_count, loaded.query_count),
        )
        workload = replace(workload, dimension=reduced.dimension, corpus_size=reduced.corpus_size)
        corpus, queries = reduced.train, reduced.test

    benchmark = VectorBenchmark(workload, corpus, queries)
    spec = workload.table_spec()

    sides: list[tuple[str, Any, list[tuple[IndexSpec, dict[str, Any]]]]] = []
    for name, dsn, entry in (
        (args.system_a, args.dsn_a, entry_a),
        (args.system_b, args.dsn_b, entry_b),
    ):
        adapter = get_adapter(name).build(dsn=dsn)
        adapter.prepare()
        adapter.start()
        adapter.wait_ready()
        adapter.load_dataset(spec, corpus)
        side_workload = entry.workload
        if not isinstance(side_workload, VectorWorkload):  # already refused above
            raise ConfigError(
                f"`{entry.id}` is not a vector benchmark",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        configurations = VectorBenchmark(side_workload, corpus, queries).configurations()
        sides.append((name, adapter, configurations))

    print(f"# Head to head: {args.system_a} vs {args.system_b}")
    print()
    print(
        f"`{args.benchmark_a}` against `{args.benchmark_b}`, interleaved query by "
        f"query with the order alternating."
    )
    print()

    probes = [benchmark._query(i) for i in range(len(queries))]
    try:
        for (index_a, search_a), (index_b, search_b) in zip(sides[0][2], sides[1][2], strict=False):
            labels: list[str] = []
            for (name, adapter, _), index, search in (
                (sides[0], index_a, search_a),
                (sides[1], index_b, search_b),
            ):
                try:
                    adapter.drop_indexes(spec)
                    adapter.build_index(spec, index)
                    adapter.set_search_parameters(search)
                except Exception as exc:  # a refusal or a rejected knob, either way no pair
                    labels.append(f"{name} refused: {exc}")
            if labels:
                print(f"- skipped — {'; '.join(labels)}")
                continue

            result = interleave(
                (sides[0][0], sides[0][1]), (sides[1][0], sides[1][1]), queries=probes
            )

            # Quality first. A latency verdict between two operating points of
            # different quality compares the operating points, not the systems --
            # and the faster one is simply the one doing less work.
            order = sorted(result.latency_a)
            recall_a = benchmark._recall([list(result.returned_a[q]) for q in order], len(order))
            recall_b = benchmark._recall([list(result.returned_b[q]) for q in order], len(order))
            label_a = f"{index_a.label()} {_render_search(search_a)}"
            label_b = f"{index_b.label()} {_render_search(search_b)}"
            print(
                print(
                    f"- `{label_a}` (recall {recall_a:.4f}) vs `{label_b}` (recall {recall_b:.4f})"
                    if recall_a is not None and recall_b is not None
                    else f"- `{label_a}` vs `{label_b}` (recall unavailable on one side)"
                )
            )

            if recall_a is None or recall_b is None or abs(recall_a - recall_b) > RECALL_TOLERANCE:
                print(
                    f"  no verdict — the two points differ in quality by more than "
                    f"{RECALL_TOLERANCE:.2f} recall. Comparing them would report the "
                    f"one doing less work as faster."
                )
                continue

            verdict = render_paired_verdict(
                result.name_a,
                result.latency_a,
                result.name_b,
                result.latency_b,
                metric="latency_ms",
            )
            dropped = (
                f" ({len(result.dropped)} query pair(s) dropped: one side did not answer)"
                if result.dropped
                else ""
            )
            print(f"  {verdict}{dropped}")

            # Within tolerance is not identical, and the side at lower recall is
            # doing less work. Saying which one it is costs a line and stops a
            # reader crediting the difference entirely to the engine.
            gap = recall_a - recall_b
            if abs(gap) > 1e-9:
                lower = sides[0][0] if gap < 0 else sides[1][0]
                print(
                    f"  caveat: {lower} operated at {abs(gap):.4f} lower recall, so part "
                    f"of its latency advantage is work it did not do."
                )
    finally:
        for _, adapter, _ in sides:
            adapter.stop()
            adapter.cleanup()

    print()
    print(
        "Interleaved: query *i* went to both systems back to back with the order "
        "alternating, so drift in the machine affects both sides equally and "
        "neither always pays the cold cache. This is the confounder a comparison "
        "of two sequential runs leaves in."
    )
    return EXIT_OK


#: How far two operating points' recall may differ and still be compared. Beyond
#: it a latency verdict describes the quality gap rather than the systems.
RECALL_TOLERANCE: Final[float] = 0.01


def _render_search(search: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(search.items())) or "(no sweep)"


def _require_comparable(a: Any, b: Any) -> None:
    """Refuse two workloads that measure different experiments.

    Index configuration may and must differ between engines. The corpus, the
    query set, k and the metric may not: comparing across them would compare the
    experiments rather than the systems, and would do it while looking like a
    head-to-head.
    """
    mismatched = [
        field
        for field in ("corpus_size", "dimension", "query_count", "k", "metric", "seed")
        if getattr(a, field) != getattr(b, field)
    ]
    if mismatched:
        detail = ", ".join(
            f"{field}: {getattr(a, field)} vs {getattr(b, field)}" for field in mismatched
        )
        raise ConfigError(
            f"the two benchmarks do not measure the same experiment ({detail}). "
            f"Index configuration may differ between engines -- their knobs are not "
            f"the same knobs -- but the corpus, queries, k and metric may not.",
            context=ErrorContext(phase=Phase.PREFLIGHT),
        )


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
    run.add_argument(
        "--dataset",
        default=None,
        help="measure a verified dataset instead of the seeded synthetic corpus",
    )
    run.add_argument("--manifest-dir", type=Path, default=None)
    run.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="render the report for an existing run")
    report.add_argument("run_dir", type=Path)
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser("compare", help="compare existing runs")
    compare.add_argument("run_dirs", type=Path, nargs="+")
    compare.set_defaults(func=cmd_compare)

    head2head = subparsers.add_parser(
        "head2head",
        help="measure two systems interleaved, query by query, and test the difference",
    )
    head2head.add_argument("--system-a", required=True, choices=sorted(ADAPTERS))
    head2head.add_argument("--dsn-a", required=True)
    head2head.add_argument("--benchmark-a", required=True, choices=sorted(BENCHMARKS))
    head2head.add_argument("--system-b", required=True, choices=sorted(ADAPTERS))
    head2head.add_argument("--dsn-b", required=True)
    head2head.add_argument(
        "--benchmark-b",
        required=True,
        choices=sorted(BENCHMARKS),
        help="may differ from --benchmark-a: two engines need different index "
        "configurations to reach the same quality, and their knobs are not the "
        "same knobs. The corpus, queries, k and metric must match.",
    )
    head2head.add_argument("--dataset", default=None)
    head2head.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    head2head.add_argument("--manifest-dir", type=Path, default=None)
    head2head.set_defaults(func=cmd_head2head)

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
