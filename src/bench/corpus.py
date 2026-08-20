"""How a corpus reaches the measurement, when it may not fit in memory.

A vector benchmark does exactly two things with its corpus: it computes the
oracle over it, and it hands it to the adapter. Both operations assumed an array,
which is the right assumption while the corpus fits — and 20 000 000 x 128
float32 is 10.2 GB on a host that is also running the database under test, so it
stops fitting well before a billion.

The alternative was an `isinstance` branch inside the benchmark, at both call
sites. That works and it is what this module exists to avoid: the benchmark has no
stake in which corpus shape it was given, and encoding the choice twice in
measurement code means a third shape reopens the measurement code.

So the corpus is bound behind one abstraction with two implementations. The
oracle equivalence between them is tested rather than assumed
(`tests/test_corpus_binding.py`): if the two disagreed, a recall figure would
depend on whether the corpus happened to fit in RAM, which is the kind of
dependency that makes published numbers unreproducible.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from theodb_bench.adapters.base import LoadOutcome
from theodb_bench.analysis.quality import (
    brute_force_ground_truth,
    distances_to_gathered,
    neighbors_ground_truth,
)
from theodb_bench.streaming import CorpusSource, neighbour_vectors, streaming_ground_truth

FloatArray = npt.NDArray[np.floating[Any]]


@runtime_checkable
class CorpusBinding(Protocol):
    """A corpus, however it is stored, as the measurement needs it."""

    @property
    def row_count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def ground_truth(
        self, queries: FloatArray, k: int, metric: str
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Exact nearest neighbours, computed by us over the bytes the system got."""
        ...

    def returned_distances(
        self, queries: FloatArray, ids: npt.NDArray[np.integer[Any]], k: int, metric: str
    ) -> npt.NDArray[np.float64]:
        """True distances to the ids the system answered with, for recall."""
        ...

    def load(self, adapter: Any, spec: Any) -> LoadOutcome:
        """Hand the corpus to the adapter by whichever path suits its shape."""
        ...

    def subset(self, rows: npt.NDArray[np.integer[Any]]) -> CorpusBinding:
        """A binding over just these rows, for a filtered oracle.

        Returned as a binding rather than an array so a streamed corpus can
        answer without materialising: the rows of one tenant out of a
        20 000 000-vector corpus are still millions of vectors.
        """
        ...


class ResidentCorpus:
    """A corpus held whole in an array."""

    def __init__(self, vectors: FloatArray) -> None:
        self._vectors = vectors

    @property
    def vectors(self) -> FloatArray:
        return self._vectors

    @property
    def row_count(self) -> int:
        return int(np.asarray(self._vectors).shape[0])

    @property
    def dimension(self) -> int:
        return int(np.asarray(self._vectors).shape[1])

    def ground_truth(
        self, queries: FloatArray, k: int, metric: str
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        ids, distances = brute_force_ground_truth(self._vectors, queries, k, metric)
        return np.asarray(ids, dtype=np.int64), distances

    def returned_distances(
        self, queries: FloatArray, ids: npt.NDArray[np.integer[Any]], k: int, metric: str
    ) -> npt.NDArray[np.float64]:
        return neighbors_ground_truth(self._vectors, queries, ids, k, metric)

    def load(self, adapter: Any, spec: Any) -> LoadOutcome:
        outcome: LoadOutcome = adapter.load_dataset(spec, self._vectors)
        return outcome

    def subset(self, rows: npt.NDArray[np.integer[Any]]) -> CorpusBinding:
        return ResidentCorpus(np.asarray(self._vectors)[rows])


class StreamedCorpus:
    """A corpus read in row ranges, never assembled into one array."""

    def __init__(self, source: CorpusSource, chunk_rows: int | None = None) -> None:
        self._source = source
        self._chunk_rows = chunk_rows

    @property
    def source(self) -> CorpusSource:
        return self._source

    @property
    def row_count(self) -> int:
        return self._source.row_count

    @property
    def dimension(self) -> int:
        return self._source.dimension

    def ground_truth(
        self, queries: FloatArray, k: int, metric: str
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        return streaming_ground_truth(
            self._source, queries, k, metric=metric, chunk_rows=self._chunk_rows
        )

    def returned_distances(
        self, queries: FloatArray, ids: npt.NDArray[np.integer[Any]], k: int, metric: str
    ) -> npt.NDArray[np.float64]:
        """Fetch only the rows the ids name, then apply the shared metric maths.

        A run at 20 000 000 vectors cannot re-read the corpus to score one
        repetition, and it does not have to: what recall needs is the distance to
        each *returned* id, so only those rows are read — distinct ones once.
        """
        selected = np.asarray(ids)[:, :k]
        gathered, _ = neighbour_vectors(self._source, selected)
        return distances_to_gathered(gathered, queries, metric)

    def subset(self, rows: npt.NDArray[np.integer[Any]]) -> CorpusBinding:
        # Gathered rather than streamed: a tenant's rows are scattered through
        # the corpus, so a range read cannot express them. `neighbour_vectors`
        # already reads exactly the rows named, distinct ones once.
        gathered, _ = neighbour_vectors(self._source, np.asarray(rows).reshape(1, -1))
        return ResidentCorpus(gathered[0])

    def load(self, adapter: Any, spec: Any) -> LoadOutcome:
        outcome: LoadOutcome = (
            adapter.load_dataset_streaming(spec, self._source)
            if self._chunk_rows is None
            else adapter.load_dataset_streaming(spec, self._source, chunk_rows=self._chunk_rows)
        )
        return outcome


def binding_for(corpus: FloatArray | CorpusSource, chunk_rows: int | None = None) -> CorpusBinding:
    """Choose the binding from what the corpus *is*, not from where it came from.

    The source case is recognised through the `CorpusSource` protocol rather than
    a concrete reader class: the reader for one file format must not be the only
    corpus this harness can stream.
    """
    if isinstance(corpus, CorpusSource):
        return StreamedCorpus(corpus, chunk_rows=chunk_rows)
    if isinstance(corpus, np.ndarray) or hasattr(corpus, "__array__"):
        return ResidentCorpus(np.asarray(corpus))
    raise TypeError(
        f"a corpus must be an array or a CorpusSource that reads rows in ranges; "
        f"got {type(corpus).__name__}"
    )
