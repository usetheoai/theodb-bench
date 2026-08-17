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
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


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
