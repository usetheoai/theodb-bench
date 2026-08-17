"""Reading a corpus that does not fit in memory.

Two things stand between this harness and a billion-vector measurement, and
neither of them is the database.

`load_dataset(spec, vectors)` takes an array, so the whole corpus has to be
resident: 1e9 x 128 float32 is 512 GB of RAM. And brute-force ground truth is a
Q x N product, which at a billion rows and ten thousand queries is 1e13 distance
computations for a single run.

Both have the same answer: read only what is needed, when it is needed. A load
streams in chunks, and the oracle reads the k x Q neighbour vectors named by the
dataset's published neighbour ids instead of the whole corpus. Published distances
are still never read — they carry someone else's precision and metric convention —
so the distances are recomputed from the vectors that were fetched.

What a billion costs, measured rather than estimated: 512 GB of raw float32,
520 GB in a `vector(128)` table, roughly 780 GB with an HNSW index, and 4.7 hours
of load at the binary-COPY rate this harness now reaches. The host this was written
on had 284 GB free, so the capability is here and the run needs a bigger machine.
Saying that is the point; a benchmark whose scale claims outrun its measurements is
worse than one whose limits are written down.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from theodb_bench.analysis.quality import pairwise_distances

#: Budget for the working distance matrix of the streaming oracle. The chunk width
#: is derived from it and the query count, so the resident set stays flat however
#: large the corpus is.
_ORACLE_CHUNK_BYTES: Final[int] = 256 * 1024 * 1024


@runtime_checkable
class CorpusSource(Protocol):
    """A corpus that can be read in ranges without being held whole."""

    @property
    def row_count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def rows(self, start: int, stop: int) -> npt.NDArray[np.floating]:
        """Rows `[start, stop)`. Only this slice need be resident."""
        ...


class PrefixSource:
    """The first `rows` records of another source, as a source.

    A workload declares a corpus size and the file may hold more. Reducing an
    array to fit is a slice; reducing a source has to be a view, because
    materialising it in order to cut it defeats the reason it is a source.

    Reads past the limit are refused even though the underlying rows exist —
    especially because they exist. A silent read past the declared corpus would
    score against vectors the run states it never loaded, so recall would be
    measured against the wrong oracle while every artifact named the right one.
    """

    def __init__(self, source: CorpusSource, rows: int) -> None:
        if rows < 1:
            raise ValueError(f"a prefix must keep at least 1 row, got {rows}")
        available = source.row_count
        if rows > available:
            raise ValueError(f"cannot take {rows} rows: the source has only {available}")
        self._source = source
        self._rows = int(rows)

    @property
    def row_count(self) -> int:
        return self._rows

    @property
    def dimension(self) -> int:
        return self._source.dimension

    def rows(self, start: int, stop: int) -> npt.NDArray[np.floating[Any]]:
        if start < 0 or stop > self._rows or stop < start:
            raise ValueError(f"rows({start}, {stop}) is outside the prefix of {self._rows} rows")
        return self._source.rows(start, stop)


def chunk_source(
    source: CorpusSource, chunk_rows: int
) -> Iterator[tuple[int, npt.NDArray[np.floating]]]:
    """Yield `(start_row_id, block)` pairs covering the corpus once.

    The starting id travels with the block rather than being counted by the
    caller: a load that is retried or resumed from the middle would otherwise
    renumber every row after the break, and the ids are what the dataset's
    neighbour lists refer to.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be at least 1, got {chunk_rows}")
    total = source.row_count
    for start in range(0, total, chunk_rows):
        yield start, source.rows(start, min(start + chunk_rows, total))


def neighbour_vectors(
    source: CorpusSource, neighbour_ids: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.floating], int]:
    """Fetch just the vectors the published neighbour ids name.

    Returns `(gathered, rows_read)` where `gathered` has shape
    `(queries, k, dimension)` and `rows_read` counts the *distinct* corpus rows
    fetched. Queries share neighbours, so reading each distinct row once keeps the
    cost proportional to distinct neighbours rather than to k times the query
    count — and `rows_read` is reported so a run can state what it actually read
    instead of implying it read the corpus.

    A neighbour id outside the corpus is refused. It happens when a published
    dataset is subsampled without remapping its neighbour lists, and dropping such
    ids quietly would raise recall by removing the neighbours a system failed to
    find.
    """
    ids = np.asarray(neighbour_ids)
    if ids.ndim != 2:
        raise ValueError(f"expected 2-D neighbour ids, got shape {ids.shape}")

    total = source.row_count
    out_of_range = ids[(ids < 0) | (ids >= total)]
    if out_of_range.size:
        raise ValueError(
            f"{out_of_range.size} neighbour id(s) outside the corpus of {total} rows "
            f"(first: {int(out_of_range.flat[0])}). A dataset subsampled without "
            f"remapping its neighbour lists points at rows that no longer exist, and "
            f"dropping them would raise recall by removing the neighbours a system "
            f"failed to find."
        )

    distinct = np.unique(ids)
    lookup = {int(row_id): position for position, row_id in enumerate(distinct)}

    # Contiguous runs are read in one call: neighbour lists cluster, and a read per
    # id would turn one sequential pass into thousands of seeks.
    fetched = np.empty((distinct.size, source.dimension), dtype=np.float32)
    position = 0
    for start, stop in _contiguous_runs(distinct):
        block = np.asarray(source.rows(start, stop), dtype=np.float32)
        fetched[position : position + block.shape[0]] = block
        position += block.shape[0]

    gathered = np.empty((ids.shape[0], ids.shape[1], source.dimension), dtype=np.float32)
    for query_index in range(ids.shape[0]):
        for neighbour_index, row_id in enumerate(ids[query_index]):
            gathered[query_index, neighbour_index] = fetched[lookup[int(row_id)]]

    return gathered, int(distinct.size)


def streaming_ground_truth(
    source: CorpusSource,
    queries: npt.NDArray[np.floating[Any]],
    k: int,
    metric: str = "l2",
    chunk_rows: int | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Exact nearest neighbours over a corpus that is never resident.

    `brute_force_ground_truth` chunks over *queries* and holds the corpus, which
    is the right trade when the corpus fits: 20 000 000 x 128 float32 is 10.2 GB,
    on a host that also runs the database under test. Here the *corpus* is what
    streams, and the top-k is carried across chunks.

    The answer is identical to the resident oracle's, and that equivalence is the
    only reason this is usable — a second oracle that disagreed with the first
    would make every recall figure depend on which one ran.

    What makes it exact is the ordering. Both oracles rank by distance and break
    ties by **ascending id**, which is a total order because ids are unique, and
    the top-k of a total order is recoverable by taking the top-k of each
    partition and merging them. A running top-k that kept only "the k smallest by
    value" per chunk would not be: when a chunk holds more ties than k, which of
    them it keeps is arbitrary, and the id-minimal ones can be dropped before the
    merge ever sees them. So ties at the selection boundary are re-admitted per
    chunk, exactly as the resident oracle re-admits them per query.
    """
    total = source.row_count
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if k > total:
        raise ValueError(f"k={k} exceeds the corpus size {total}")

    query_array = np.asarray(queries)
    query_count = int(query_array.shape[0])
    if chunk_rows is None:
        # float64 distances plus the boolean tie mask over the same shape.
        chunk_rows = max(k, _ORACLE_CHUNK_BYTES // max(1, query_count * 9))

    best_distances: npt.NDArray[np.float64] = np.empty((query_count, 0), dtype=np.float64)
    best_ids: npt.NDArray[np.int64] = np.empty((query_count, 0), dtype=np.int64)

    for start, block in chunk_source(source, chunk_rows):
        distances = np.asarray(pairwise_distances(block, query_array, metric), dtype=np.float64)
        ids = start + np.arange(block.shape[0], dtype=np.int64)
        chunk_distances, chunk_ids = _select_top_k(distances, ids, k)
        best_distances, best_ids = _merge_top_k(
            best_distances, best_ids, chunk_distances, chunk_ids, k
        )

    return best_ids, best_distances


def _select_top_k(
    distances: npt.NDArray[np.float64], ids: npt.NDArray[np.int64], k: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """The k best of one chunk, ordered by distance then ascending id."""
    width = distances.shape[1]
    take = min(k, width)

    if take == width:
        order = np.lexsort((np.broadcast_to(ids, distances.shape), distances), axis=1)
        selected = order[:, :take]
    else:
        partitioned = np.argpartition(distances, take - 1, axis=1)[:, :take]
        partitioned_distances = np.take_along_axis(distances, partitioned, 1)
        thresholds = partitioned_distances.max(axis=1, keepdims=True)
        order = np.lexsort((ids[partitioned], partitioned_distances), axis=1)
        selected = np.take_along_axis(partitioned, order, 1)

        # Rows where more candidates tie with the k-th than fit: the partition
        # kept an arbitrary subset of the tie, so those rows are redone against
        # every tied candidate. Real corpora hit this rarely; a corpus with
        # duplicate vectors hits it on every row, which is why it is not skipped.
        tied = (distances <= thresholds).sum(axis=1) > take
        for row in np.flatnonzero(tied):
            candidates = np.flatnonzero(distances[row] <= thresholds[row, 0])
            row_distances = distances[row, candidates]
            selected[row] = candidates[np.lexsort((ids[candidates], row_distances))][:take]

    return np.take_along_axis(distances, selected, 1), ids[selected]


def _merge_top_k(
    best_distances: npt.NDArray[np.float64],
    best_ids: npt.NDArray[np.int64],
    chunk_distances: npt.NDArray[np.float64],
    chunk_ids: npt.NDArray[np.int64],
    k: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Merge two per-query top-k sets, keeping the k best of the union."""
    distances = np.concatenate((best_distances, chunk_distances), axis=1)
    ids = np.concatenate((best_ids, chunk_ids), axis=1)
    take = min(k, distances.shape[1])
    order = np.lexsort((ids, distances), axis=1)[:, :take]
    return np.take_along_axis(distances, order, 1), np.take_along_axis(ids, order, 1)


def _contiguous_runs(sorted_ids: npt.NDArray[np.integer]) -> Iterator[tuple[int, int]]:
    """Maximal `[start, stop)` runs of consecutive ids in a sorted array."""
    if sorted_ids.size == 0:
        return
    start = previous = int(sorted_ids[0])
    for value in sorted_ids[1:]:
        current = int(value)
        if current != previous + 1:
            yield start, previous + 1
            start = current
        previous = current
    yield start, previous + 1
