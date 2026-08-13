"""Same data, same queries, three paths. A wrong answer is never a fast answer."""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import AnalyticalQuery
from theodb_bench.adapters.fake import FakeAdapter, FakeConfig
from theodb_bench.bench.analytical import (
    COLUMNAR,
    PARQUET,
    PATHS,
    QUERIES,
    ROW,
    AnalyticalBenchmark,
    AnalyticalWorkload,
    answers_match,
    expected_answer,
    generate_rows,
    timed_reference_scan,
)
from theodb_bench.errors import ConfigError


def _workload(**overrides: object) -> AnalyticalWorkload:
    base: dict[str, object] = {"row_count": 2000, "repetitions": 2, "warmup_queries": 1}
    base.update(overrides)
    return AnalyticalWorkload(**base)  # type: ignore[arg-type]


def _ready(config: FakeConfig | None = None) -> FakeAdapter:
    adapter = FakeAdapter(config)
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    return adapter


# ------------------------------------------------------------------- workload


def test_unknown_paths_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown execution path"):
        _workload(paths=("magnetic-tape",))


def test_a_degenerate_workload_is_refused() -> None:
    with pytest.raises(ConfigError, match="row_count"):
        _workload(row_count=0)
    with pytest.raises(ConfigError, match="repetitions"):
        _workload(repetitions=0)


def test_rows_are_deterministic_for_a_seed() -> None:
    assert generate_rows(_workload(seed=6)) == generate_rows(_workload(seed=6))


def test_every_path_receives_identical_rows() -> None:
    # The equality is the comparison: three paths measured on different data
    # would measure the data.
    benchmark = AnalyticalBenchmark(_workload())
    adapter = _ready()
    try:
        for path in PATHS:
            benchmark.load(adapter, path)
        stored = [adapter._analytical[benchmark.workload.table_for(p).name] for p in PATHS]
        assert stored[0] == stored[1] == stored[2]
    finally:
        adapter.stop()


# --------------------------------------------------------------------- oracle


def test_the_oracle_counts_rows() -> None:
    rows = [(0, 1.0, "a", 1), (1, 2.0, "b", 2)]
    assert expected_answer(rows, "total_rows") == ((2,),)


def test_the_oracle_sums_a_column() -> None:
    rows = [(0, 1.5, "a", 1), (1, 2.5, "b", 2)]
    assert expected_answer(rows, "sum_amount") == ((4.0,),)


def test_the_oracle_groups_by_key() -> None:
    rows = [(0, 1.0, "a", 1), (1, 2.0, "b", 2), (2, 3.0, "a", 3)]
    assert expected_answer(rows, "group_by_category") == (("a", 4.0), ("b", 2.0))


def test_the_oracle_applies_both_filter_predicates() -> None:
    rows = [(0, 5.0, "a", 1), (1, -5.0, "a", 2), (2, 7.0, "b", 3)]
    assert expected_answer(rows, "filtered_sum") == ((5.0,),)


def test_an_unknown_query_has_no_oracle_and_says_so() -> None:
    with pytest.raises(ConfigError, match="no oracle"):
        expected_answer([], "select_star")


# ---------------------------------------------------------------- comparison


def test_float_tolerance_is_allowed() -> None:
    assert answers_match(((1.0000001,),), ((1.0,),), tolerance=1e-6)


def test_a_difference_beyond_tolerance_is_a_wrong_answer() -> None:
    assert not answers_match(((1.1,),), ((1.0,),), tolerance=1e-6)


def test_no_tolerance_covers_a_different_row_count() -> None:
    # A missing group is a wrong answer, not a rounding difference.
    assert not answers_match((("a", 1.0),), (("a", 1.0), ("b", 2.0)), tolerance=1.0)


def test_no_tolerance_covers_a_different_key() -> None:
    assert not answers_match((("a", 1.0),), (("z", 1.0),), tolerance=1.0)


# ------------------------------------------------------------------- queries


@pytest.mark.parametrize("path", PATHS)
def test_every_path_answers_every_query_correctly(path: str) -> None:
    workload = _workload()
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        benchmark.load(adapter, path)
        for query in QUERIES:
            measurement = benchmark.run_query(adapter, path, query)
            assert measurement.status == "measured", measurement.status_detail
            assert measurement.rows_per_second is not None
    finally:
        adapter.stop()


def test_a_wrong_answer_discards_the_timing() -> None:
    # The failure that is easiest to miss on this surface: an aggregate is one
    # number, and one wrong number looks exactly like a right one.
    workload = _workload()
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        benchmark.load(adapter, ROW)
        benchmark.oracle["sum_amount"] = ((123456.0,),)
        measurement = benchmark.run_query(
            adapter, ROW, AnalyticalQuery(id="sum_amount", description="")
        )
        assert measurement.status == "invalid"
        assert measurement.wall_seconds is None
        assert measurement.status_detail is not None
        assert "timing was discarded" in measurement.status_detail
    finally:
        adapter.stop()


def test_stage_timings_say_where_the_time_went() -> None:
    # "The query got faster" is not a finding until it says which stage moved.
    workload = _workload()
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        benchmark.load(adapter, PARQUET)
        measurement = benchmark.run_query(adapter, PARQUET, QUERIES[1])
        assert {"metadata", "prune", "read", "aggregate"} <= set(measurement.stage_seconds)
    finally:
        adapter.stop()


def test_only_parquet_pays_a_metadata_stage() -> None:
    workload = _workload()
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        benchmark.load(adapter, ROW)
        benchmark.load(adapter, PARQUET)
        row = benchmark.run_query(adapter, ROW, QUERIES[1])
        parquet = benchmark.run_query(adapter, PARQUET, QUERIES[1])
        assert "metadata" not in row.stage_seconds
        assert parquet.stage_seconds["metadata"] > 0
    finally:
        adapter.stop()


def test_the_columnar_path_reads_fewer_bytes_than_the_row_path() -> None:
    workload = _workload()
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        benchmark.load(adapter, ROW)
        benchmark.load(adapter, COLUMNAR)
        row = benchmark.run_query(adapter, ROW, QUERIES[1])
        columnar = benchmark.run_query(adapter, COLUMNAR, QUERIES[1])
        assert row.bytes_read is not None and columnar.bytes_read is not None
        assert columnar.bytes_read < row.bytes_read
    finally:
        adapter.stop()


# -------------------------------------------------------------- full sweep


def test_the_full_run_covers_every_path_and_query() -> None:
    workload = _workload(row_count=500)
    adapter = _ready()
    try:
        measurements = AnalyticalBenchmark(workload).run(adapter)
        assert len(measurements) == len(PATHS) * len(QUERIES)
        assert all(m.status == "measured" for m in measurements)
    finally:
        adapter.stop()


def test_an_unsupported_path_appears_with_its_status_rather_than_vanishing() -> None:
    # A table that silently omits a path reads as though it was never tried.
    workload = _workload(row_count=300)
    adapter = _ready(FakeConfig(capabilities={"vector_exact": True, "columnar": True}))
    try:
        measurements = AnalyticalBenchmark(workload).run(adapter)
        parquet = [m for m in measurements if m.path == PARQUET]
        assert parquet
        assert all(m.status == "unsupported" for m in parquet)
    finally:
        adapter.stop()


def test_the_comparison_lines_paths_up_per_query() -> None:
    workload = _workload(row_count=500)
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        comparison = benchmark.compare_paths(benchmark.run(adapter))
        assert set(comparison["queries"]) == {q.id for q in QUERIES}
        for paths in comparison["queries"].values():
            assert set(paths) == set(PATHS)
        assert "describes the execution path" in comparison["note"]
    finally:
        adapter.stop()


def test_speedups_are_expressed_against_the_row_baseline() -> None:
    workload = _workload(row_count=3000)
    benchmark = AnalyticalBenchmark(workload)
    adapter = _ready()
    try:
        comparison = benchmark.compare_paths(benchmark.run(adapter))
        speedups = comparison["speedup_over_row"]
        assert speedups
        for per_path in speedups.values():
            assert ROW not in per_path
            assert all(value > 0 for value in per_path.values())
    finally:
        adapter.stop()


# ------------------------------------------------------------------ reference


def test_the_reference_scan_gives_a_floor_and_the_right_answer() -> None:
    rows = generate_rows(_workload(row_count=100))
    seconds, answer = timed_reference_scan(rows, "sum_amount")
    assert seconds > 0
    assert answer == expected_answer(rows, "sum_amount")
