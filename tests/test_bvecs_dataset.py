"""A `.bvecs` dataset reaching a run without ever being held whole.

The HDF5 path reads `train` into an array, which is right up to a few million
vectors and wrong past that: BIGANN's first 20 000 000 SIFT descriptors are
2.64 GB on disk and 10.2 GB as the float32 the oracle needs. So this dataset
hands the run a *source* and the rest of the pipeline already accepts one —
`binding_for` picks the streamed binding from the corpus itself.

Ground truth is computed here rather than read. BIGANN publishes neighbour ids,
but they index the full billion: against the first 20M they name rows that do not
exist, which is precisely the case `neighbour_vectors` refuses instead of
dropping — dropping them would raise recall by removing the neighbours a system
failed to find.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.adapters.base import (
    BuildOutcome,
    IndexSpec,
    KnnQuery,
    KnnResult,
    LoadOutcome,
    SystemAdapter,
    VectorTableSpec,
)
from theodb_bench.formats import read_bvecs_dataset
from theodb_bench.streaming import CorpusSource


def _write(path: Path, vectors: npt.NDArray[np.unsignedinteger[Any]]) -> Path:
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack("<i", vector.shape[0]))
            handle.write(vector.astype(np.uint8).tobytes())
    return path


@pytest.fixture
def pair(
    tmp_path: Path,
) -> tuple[Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]]:
    rng = np.random.default_rng(20260817)
    base = rng.integers(0, 256, size=(64, 10), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(7, 10), dtype=np.uint8)
    return (
        _write(tmp_path / "base.bvecs", base),
        _write(tmp_path / "query.bvecs", queries),
        base,
        queries,
    )


def test_the_corpus_arrives_as_a_source_and_the_queries_as_an_array(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    """Asymmetric on purpose: ten thousand queries are megabytes and every
    repetition reads all of them, while the corpus is read once per chunk."""
    base_path, query_path, base, queries = pair
    dataset = read_bvecs_dataset(base_path, query_path)

    assert isinstance(dataset.train, CorpusSource)
    assert isinstance(dataset.test, np.ndarray)
    assert dataset.corpus_size == base.shape[0]
    assert dataset.query_count == queries.shape[0]
    assert dataset.dimension == base.shape[1]


def test_no_published_neighbours_are_carried(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    """BIGANN's neighbour ids index the full billion. Carrying them against a
    20M prefix would point at rows that do not exist."""
    base_path, query_path, _, _ = pair

    assert read_bvecs_dataset(base_path, query_path).neighbors is None


def test_a_query_file_of_a_different_dimension_is_refused(tmp_path: Path) -> None:
    """The one mismatch that produces numbers instead of an error: distances
    would broadcast or fail deep inside the oracle."""
    rng = np.random.default_rng(2)
    base = _write(tmp_path / "b.bvecs", rng.integers(0, 256, size=(8, 10), dtype=np.uint8))
    query = _write(tmp_path / "q.bvecs", rng.integers(0, 256, size=(3, 12), dtype=np.uint8))

    with pytest.raises(Exception, match="dimension"):
        read_bvecs_dataset(base, query)


def test_subsampling_takes_a_prefix_of_both(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    base_path, query_path, base, queries = pair

    reduced = read_bvecs_dataset(base_path, query_path).subsample(20, 3)

    assert reduced.corpus_size == 20
    assert reduced.query_count == 3
    np.testing.assert_array_equal(reduced.train.rows(0, 20), base[:20].astype(np.float32))
    np.testing.assert_array_equal(reduced.test, queries[:3].astype(np.float32))


def test_subsampling_beyond_the_file_is_refused(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    base_path, query_path, base, _ = pair

    with pytest.raises(Exception, match="cannot take"):
        read_bvecs_dataset(base_path, query_path).subsample(base.shape[0] + 1, 3)


def test_a_run_over_a_bvecs_dataset_measures_the_streamed_corpus(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    """End to end at toy scale: the benchmark builds, streams its load, and
    scores recall — the same path a 20M run takes."""
    from theodb_bench.bench.corpus import StreamedCorpus
    from theodb_bench.bench.vector import VectorWorkload

    base_path, query_path, base, _ = pair
    dataset = read_bvecs_dataset(base_path, query_path).subsample(40, 5)
    workload = VectorWorkload(corpus_size=40, dimension=10, query_count=5, k=4)

    benchmark = workload.build(dataset.train, dataset.test)

    assert isinstance(benchmark.binding, StreamedCorpus)
    # The oracle ran over the streamed corpus, and agrees with the resident one.
    from theodb_bench.analysis.quality import brute_force_ground_truth

    reference, _ = brute_force_ground_truth(base[:40].astype(np.float32), dataset.test, 4)
    np.testing.assert_array_equal(benchmark._ground_truth_ids, reference)


def test_asking_a_streamed_benchmark_for_its_corpus_array_is_refused(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    """The attribute exists for resident runs. Materialising 20M x 128 float32 to
    satisfy an attribute access is 10.2 GB — the exact allocation being avoided."""
    from theodb_bench.bench.vector import VectorWorkload

    base_path, query_path, _, _ = pair
    dataset = read_bvecs_dataset(base_path, query_path).subsample(40, 5)
    benchmark = VectorWorkload(corpus_size=40, dimension=10, query_count=5, k=4).build(
        dataset.train, dataset.test
    )

    with pytest.raises(Exception, match="streams its corpus"):
        _ = benchmark.corpus


# --------------------------------------------------- the registered 20M suites


def test_the_reference_scale_is_registered_and_declares_twenty_million() -> None:
    """The scale chosen from measured size: 1.27 GB per million on the measured
    host puts 20M at 25.4 GB, which is 9% of that disk."""
    from theodb_bench.bench.vector import VectorWorkload
    from theodb_bench.registry import BENCHMARKS

    for name in ("vector/bigann20m/hnsw", "vector/bigann20m/load"):
        workload = BENCHMARKS[name].workload
        assert isinstance(workload, VectorWorkload)
        assert workload.corpus_size == 20_000_000
        assert workload.dimension == 128


def test_the_load_only_suite_builds_no_index() -> None:
    """`--index none` is not a knob, so isolating the load from the build needs
    its own entry: minutes of work against hours."""
    from theodb_bench.bench.vector import VectorWorkload
    from theodb_bench.registry import BENCHMARKS

    workload = BENCHMARKS["vector/bigann20m/load"].workload
    assert isinstance(workload, VectorWorkload)
    assert [index.kind for index in workload.indexes] == ["none"]


def test_the_twenty_million_dataset_manifest_matches_the_suite() -> None:
    """A manifest declaring a different corpus size than the suite would run a
    prefix while every artifact named the whole."""
    from pathlib import Path

    from theodb_bench.bench.vector import VectorWorkload
    from theodb_bench.datasets import DatasetManifest
    from theodb_bench.registry import BENCHMARKS

    manifest = DatasetManifest.load(
        Path(__file__).resolve().parents[1] / "datasets/manifests/bigann-20m-euclidean.json"
    )
    workload = BENCHMARKS["vector/bigann20m/hnsw"].workload
    assert isinstance(workload, VectorWorkload)
    properties = manifest.properties
    assert properties is not None

    assert properties["corpus_size"] == workload.corpus_size
    assert properties["dimension"] == workload.dimension
    assert properties["metric"] == workload.metric
    # No published neighbours: they index the full billion.
    assert properties["neighbours_per_query"] == 0


def test_the_manifest_declares_both_a_corpus_and_a_query_file() -> None:
    """Queries sliced out of the corpus would each be their own nearest
    neighbour, raising recall by one hit per query for every system alike."""
    from pathlib import Path

    from theodb_bench.datasets import DatasetManifest

    manifest = DatasetManifest.load(
        Path(__file__).resolve().parents[1] / "datasets/manifests/bigann-20m-euclidean.json"
    )

    assert manifest.file_by_role("train") is not None
    assert manifest.file_by_role("queries") is not None


def test_the_corpus_file_size_is_exactly_twenty_million_records() -> None:
    """132 bytes per record (int32 dimension + 128 uint8), verified against the
    real file's first bytes. A size that is not a multiple is a truncated fetch."""
    from pathlib import Path

    from theodb_bench.datasets import DatasetManifest

    manifest = DatasetManifest.load(
        Path(__file__).resolve().parents[1] / "datasets/manifests/bigann-20m-euclidean.json"
    )
    entry = manifest.file_by_role("train")
    assert entry is not None

    assert entry.size_bytes == 20_000_000 * (4 + 128)


# ------------------------------------- the path a real run takes, end to end
#
# Measured 2026-08-17: a 20 000 000-vector run reached the measurement window and
# then refused, because `_recall` validated returned ids against `self.corpus`
# rather than the binding — 31 minutes to find a one-line defect the suite should
# have caught in milliseconds.
#
# It did not, and the reason is worth more than the fix: the test above builds the
# benchmark and checks the oracle, then stops. It asserted the *setup* and called
# it the path. Recall is computed inside `measure`, which nothing exercised on a
# streamed corpus.


class _RecordingAdapter(SystemAdapter):
    """Answers every query with the same ids, so recall is computed for real.

    A real subclass rather than a duck-typed stand-in: the adapter contract is an
    ABC, and a double that cannot satisfy it is not evidence that the path works.
    """

    system_id = "fake"

    def __init__(self, ids: tuple[int, ...]) -> None:
        self._ids = ids
        self.streamed = 0

    def capabilities(self) -> dict[str, bool]:
        return {"vector_exact": True}

    def prepare(self) -> None:
        return None

    def start(self) -> None:
        return None

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        return None

    def load_dataset(self, spec: VectorTableSpec, vectors: Any) -> LoadOutcome:
        raise AssertionError("the resident load path was taken for a streamed corpus")

    def load_dataset_streaming(
        self, spec: VectorTableSpec, source: Any, *, chunk_rows: int = 50_000
    ) -> LoadOutcome:
        self.streamed += 1
        rows = int(source.row_count)
        return LoadOutcome(seconds=0.1, rows_loaded=rows, rows_expected=rows)

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})

    def execute(self, query: KnnQuery) -> KnnResult:
        return KnnResult(
            ids=self._ids, distances=tuple(0.0 for _ in self._ids), latency_seconds=0.001
        )

    def collect_stats(self) -> dict[str, Any]:
        return {}

    def export_config(self) -> dict[str, Any]:
        return {}

    def stop(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


def test_a_streamed_run_computes_recall_instead_of_refusing(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    """The whole point of the streamed path: a measured window that ends in a
    number. Validating ids against an array the run deliberately never holds
    turned that into a refusal after the work was already done."""
    from theodb_bench.bench.vector import VectorWorkload

    base_path, query_path, _, _ = pair
    dataset = read_bvecs_dataset(base_path, query_path).subsample(40, 5)
    workload = VectorWorkload(corpus_size=40, dimension=10, query_count=5, k=3)
    benchmark = workload.build(dataset.train, dataset.test)
    adapter = _RecordingAdapter(ids=(0, 1, 2))

    benchmark.load(adapter)
    result = benchmark.measure(adapter, repetition=1)

    assert adapter.streamed == 1, "the load did not go through the streaming path"
    assert result.successes == 5
    assert result.recall is not None, "a streamed run must produce a recall figure"
    assert 0.0 <= result.recall <= 1.0


def test_an_id_outside_a_streamed_corpus_is_still_a_correctness_failure(
    pair: tuple[
        Path, Path, npt.NDArray[np.unsignedinteger[Any]], npt.NDArray[np.unsignedinteger[Any]]
    ],
) -> None:
    """The check `_recall` was doing when it reached for the array. It has to
    survive the fix: a system answering with an id it was never given is wrong,
    and scoring it would reward the wrong answer."""
    from theodb_bench.bench.vector import VectorWorkload
    from theodb_bench.errors import MeasurementError

    base_path, query_path, _, _ = pair
    dataset = read_bvecs_dataset(base_path, query_path).subsample(40, 5)
    benchmark = VectorWorkload(corpus_size=40, dimension=10, query_count=5, k=3).build(
        dataset.train, dataset.test
    )
    adapter = _RecordingAdapter(ids=(0, 1, 999))

    benchmark.load(adapter)

    with pytest.raises(MeasurementError, match="outside the corpus"):
        benchmark.measure(adapter, repetition=1)
