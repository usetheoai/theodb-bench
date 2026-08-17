"""How a corpus reaches the measurement, when it may not fit in memory.

`VectorBenchmark` did two things with its corpus: computed the oracle over it and
handed it to the adapter. Both assumed an array. A 20 000 000-vector corpus is
10.2 GB as float32 on a host that also runs the database under test, so it
arrives as a source that is read in ranges instead.

The branch could have lived inside the benchmark — `if isinstance(corpus,
np.ndarray)` around both call sites, twice. It lives here instead, as two
implementations of one binding, because the benchmark has no business knowing
which one it got: adding a third corpus shape later must not reopen the
measurement code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.bench.corpus import (
    CorpusBinding,
    ResidentCorpus,
    StreamedCorpus,
    binding_for,
)
from theodb_bench.streaming import CorpusSource


class _Source:
    def __init__(self, array: npt.NDArray[np.floating[Any]]) -> None:
        self._array = array
        self.reads: list[tuple[int, int]] = []

    @property
    def row_count(self) -> int:
        return int(self._array.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._array.shape[1])

    def rows(self, start: int, stop: int) -> npt.NDArray[np.floating[Any]]:
        self.reads.append((start, stop))
        return self._array[start:stop]


class _Adapter:
    """Records which load entry point was taken."""

    def __init__(self) -> None:
        self.resident_calls = 0
        self.streaming_calls = 0

    def load_dataset(self, spec: Any, vectors: Any) -> Any:
        self.resident_calls += 1
        return _Outcome(int(np.asarray(vectors).shape[0]))

    def load_dataset_streaming(self, spec: Any, source: Any, chunk_rows: int = 0) -> Any:
        self.streaming_calls += 1
        return _Outcome(source.row_count)


class _Outcome:
    def __init__(self, rows: int) -> None:
        self.rows_loaded = rows
        self.rows_expected = rows
        self.complete = True
        self.seconds = 1.0


CORPUS = np.random.default_rng(20260817).random((120, 6), dtype=np.float32)
QUERIES = np.random.default_rng(4).random((5, 6), dtype=np.float32)


def test_both_bindings_satisfy_the_protocol() -> None:
    assert isinstance(ResidentCorpus(CORPUS), CorpusBinding)
    assert isinstance(StreamedCorpus(_Source(CORPUS)), CorpusBinding)


def test_an_array_binds_resident_and_a_source_binds_streamed() -> None:
    assert isinstance(binding_for(CORPUS), ResidentCorpus)
    assert isinstance(binding_for(_Source(CORPUS)), StreamedCorpus)


def test_a_source_is_recognised_by_the_protocol_not_by_its_class() -> None:
    """Anything that reads rows in ranges binds streamed: the reader for one file
    format must not be the only corpus the benchmark can stream."""
    source = _Source(CORPUS)
    assert isinstance(source, CorpusSource)
    assert isinstance(binding_for(source), StreamedCorpus)


def test_something_that_is_neither_is_refused_rather_than_assumed() -> None:
    with pytest.raises(TypeError, match="corpus"):
        binding_for("a path, perhaps")  # type: ignore[arg-type]


def test_the_two_bindings_agree_on_the_oracle() -> None:
    """The equivalence the whole design rests on. If the oracles disagreed, a
    recall figure would depend on whether the corpus happened to fit in RAM."""
    resident_ids, resident_distances = ResidentCorpus(CORPUS).ground_truth(QUERIES, 10, "l2")
    streamed_ids, streamed_distances = StreamedCorpus(_Source(CORPUS), chunk_rows=16).ground_truth(
        QUERIES, 10, "l2"
    )

    np.testing.assert_array_equal(streamed_ids, resident_ids)
    np.testing.assert_allclose(streamed_distances, resident_distances, rtol=1e-6)


def test_the_two_bindings_agree_on_shape() -> None:
    for binding in (ResidentCorpus(CORPUS), StreamedCorpus(_Source(CORPUS))):
        assert binding.row_count == 120
        assert binding.dimension == 6


def test_each_binding_takes_its_own_load_path() -> None:
    resident_adapter, streaming_adapter = _Adapter(), _Adapter()

    ResidentCorpus(CORPUS).load(resident_adapter, spec=object())
    StreamedCorpus(_Source(CORPUS)).load(streaming_adapter, spec=object())

    assert (resident_adapter.resident_calls, resident_adapter.streaming_calls) == (1, 0)
    assert (streaming_adapter.resident_calls, streaming_adapter.streaming_calls) == (0, 1)


def test_the_streamed_binding_never_materialises_the_corpus() -> None:
    """The property that makes 20M measurable: the corpus is read in bounded
    slices for the oracle, and never assembled into one array."""
    source = _Source(CORPUS)
    binding = StreamedCorpus(source, chunk_rows=30)

    binding.ground_truth(QUERIES, 5, "l2")

    assert source.reads == [(0, 30), (30, 60), (60, 90), (90, 120)]
    assert max(stop - start for start, stop in source.reads) == 30


# ------------------------------------- the distances of what the system returned
#
# Recall needs the true distance of every id the system answered with, which is a
# second read of the corpus. The resident binding gathers by fancy-indexing; the
# streamed one fetches only the distinct rows the ids name. The metric arithmetic
# after the gather is one implementation, shared — duplicating it would let the
# two bindings drift apart on a metric nobody checked.


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_the_two_bindings_agree_on_returned_distances(metric: str) -> None:
    ids = np.array([[3, 90, 7], [11, 12, 13], [0, 119, 60], [5, 5, 5], [42, 41, 40]])

    resident = ResidentCorpus(CORPUS).returned_distances(QUERIES, ids, 3, metric)
    streamed = StreamedCorpus(_Source(CORPUS)).returned_distances(QUERIES, ids, 3, metric)

    np.testing.assert_allclose(streamed, resident, rtol=1e-9, atol=1e-12)


def test_the_streamed_binding_reads_only_the_ids_it_was_given() -> None:
    """A run at 20M cannot re-read the corpus to score one repetition."""
    source = _Source(CORPUS)
    ids = np.array([[3, 4], [3, 4], [80, 81], [0, 1], [10, 11]])

    StreamedCorpus(source).returned_distances(QUERIES, ids, 2, "l2")

    assert sum(stop - start for start, stop in source.reads) == 8


def test_an_id_outside_the_corpus_is_refused_by_both_bindings() -> None:
    ids = np.array([[0, 120], [1, 2], [3, 4], [5, 6], [7, 8]])

    for binding in (ResidentCorpus(CORPUS), StreamedCorpus(_Source(CORPUS))):
        with pytest.raises(Exception, match="outside"):
            binding.returned_distances(QUERIES, ids, 2, "l2")
