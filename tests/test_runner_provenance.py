"""A manifest may not name a dataset the run did not measure.

This is the first link of the provenance chain. If it can be broken, every
artifact downstream still looks correct while describing different data, and
nothing in the bundle reveals it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.adapters.fake import FakeAdapter
from theodb_bench.bench.vector import VectorBenchmark, VectorWorkload
from theodb_bench.errors import ConfigError
from theodb_bench.runner import RunRequest, run_benchmark

DIMENSION = 8


def _workload(**overrides: object) -> VectorWorkload:
    base: dict[str, object] = {
        "corpus_size": 64,
        "dimension": DIMENSION,
        "query_count": 12,
        "k": 4,
    }
    base.update(overrides)
    return VectorWorkload(**base)  # type: ignore[arg-type]


def _vectors(rows: int, dimension: int = DIMENSION) -> npt.NDArray[np.float32]:
    return np.random.default_rng(3).standard_normal((rows, dimension)).astype(np.float32)


def _request(tmp_path: Path, **overrides: object) -> RunRequest:
    base: dict[str, object] = {
        "benchmark_id": "vector/synthetic/smoke",
        "workload": _workload(),
        "adapter_factory": FakeAdapter,
        "results_root": tmp_path / "results",
        "repetitions": 1,
    }
    base.update(overrides)
    return RunRequest(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- the defect


def test_declaring_a_dataset_without_supplying_it_is_refused(tmp_path: Path) -> None:
    # The defect this closes: the manifest said sift1m while the run measured a
    # seeded synthetic corpus, and every other artifact looked correct.
    with pytest.raises(ConfigError, match="did not measure"):
        run_benchmark(_request(tmp_path, dataset_id="sift1m", dataset_sha256="0" * 64))


def test_supplying_data_without_declaring_it_is_refused(tmp_path: Path) -> None:
    # The mirror image: measured data that the manifest cannot identify.
    with pytest.raises(ConfigError, match="must be identifiable"):
        run_benchmark(_request(tmp_path, corpus=_vectors(64), queries=_vectors(12)))


def test_a_synthetic_run_declares_no_dataset_and_is_valid(tmp_path: Path) -> None:
    outcome = run_benchmark(_request(tmp_path))
    assert outcome.status == "VALID"
    assert "dataset" not in outcome.bundle.read_artifact("manifest")


def test_a_declared_dataset_that_was_supplied_is_recorded(tmp_path: Path) -> None:
    outcome = run_benchmark(
        _request(
            tmp_path,
            corpus=_vectors(64),
            queries=_vectors(12),
            dataset_id="toy",
            dataset_version="1",
            dataset_sha256="a" * 64,
        )
    )
    manifest = outcome.bundle.read_artifact("manifest")
    assert manifest["dataset"]["id"] == "toy"
    assert manifest["dataset"]["sha256"] == "a" * 64
    assert outcome.status == "VALID"


# ------------------------------------------------------------ supplied data


def test_supplied_vectors_are_the_ones_measured(tmp_path: Path) -> None:
    corpus = _vectors(64)
    benchmark = VectorBenchmark(_workload(), corpus, _vectors(12))
    assert np.array_equal(benchmark.corpus, corpus)
    assert benchmark.synthetic is False


def test_a_generated_corpus_is_marked_synthetic() -> None:
    assert VectorBenchmark(_workload()).synthetic is True


def test_half_a_dataset_is_refused() -> None:
    with pytest.raises(ConfigError, match="both a corpus and a query set"):
        VectorBenchmark(_workload(), _vectors(64), None)


def test_a_corpus_of_the_wrong_dimension_is_refused() -> None:
    with pytest.raises(ConfigError, match="declares dimension"):
        VectorBenchmark(_workload(), _vectors(64, 16), _vectors(12, 16))


def test_a_corpus_smaller_than_k_is_refused() -> None:
    with pytest.raises(ConfigError, match="exceeds the supplied corpus"):
        VectorBenchmark(_workload(k=10), _vectors(3), _vectors(2))


def test_a_one_dimensional_corpus_is_refused() -> None:
    flat = np.zeros(8, dtype=np.float32)
    with pytest.raises(ConfigError, match="must be 2-D"):
        VectorBenchmark(_workload(), flat, flat)


def test_recall_over_supplied_vectors_is_still_exact(tmp_path: Path) -> None:
    # The oracle is computed from the supplied bytes, so an exact-search system
    # must still score 1.0 -- proving the dataset path did not bypass it.
    outcome = run_benchmark(
        _request(
            tmp_path,
            corpus=_vectors(64),
            queries=_vectors(12),
            dataset_id="toy",
            dataset_sha256="b" * 64,
        )
    )
    recalls = [r.recall for p in outcome.points for r in p.repetitions if r.recall is not None]
    assert recalls
    assert all(value == pytest.approx(1.0) for value in recalls)
