"""Loading a corpus larger than memory, and scoring recall without holding it.

Two things stand between this harness and a billion-vector measurement, and
neither is the database.

The first is the load signature: `load_dataset(spec, vectors)` takes an array, so
the whole corpus must be resident. At 1e9 x 128 float32 that is 512 GB of RAM.

The second is the oracle. Brute-force ground truth is a Q x N product; at a
billion rows and 10 000 queries that is 1e13 distance computations per run.
Published ANN datasets ship neighbour ids for exactly this reason, and
`neighbors_ground_truth` already recomputes distances from them rather than
trusting the published ones — but it takes the full corpus, when all it needs are
the k x Q neighbour vectors.

What a billion actually costs, measured on the host this was written on: 512 GB of
raw float32, 520 GB in a `vector(128)` table, about 780 GB with an HNSW index, and
4.7 hours of load at the binary-COPY rate. The host had 284 GB free. That is
recorded here because a benchmark whose scale claims are aspirational is worse
than one whose limits are stated.
"""

from __future__ import annotations

import numpy as np
import pytest
from theodb_bench.streaming import (
    CorpusSource,
    chunk_source,
    neighbour_vectors,
)


class _ArraySource:
    """A source backed by an in-memory array, standing in for a file reader."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    @property
    def row_count(self) -> int:
        return int(self._array.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._array.shape[1])

    def rows(self, start: int, stop: int) -> np.ndarray:
        return self._array[start:stop]


def test_a_source_satisfies_the_protocol() -> None:
    assert isinstance(_ArraySource(np.zeros((4, 2), dtype=np.float32)), CorpusSource)


def test_chunks_cover_every_row_exactly_once() -> None:
    source = _ArraySource(np.arange(20, dtype=np.float32).reshape(10, 2))

    seen = [row for start, block in chunk_source(source, chunk_rows=3) for row in block]

    assert len(seen) == 10
    np.testing.assert_array_equal(np.asarray(seen), source.rows(0, 10))


def test_chunks_carry_their_starting_row_id() -> None:
    """The id has to come from the chunk, not from a counter the caller keeps:
    a resumed or retried load would otherwise renumber every row after the break."""
    source = _ArraySource(np.zeros((10, 2), dtype=np.float32))

    starts = [start for start, _ in chunk_source(source, chunk_rows=4)]

    assert starts == [0, 4, 8]


def test_no_chunk_is_larger_than_asked() -> None:
    source = _ArraySource(np.zeros((10, 3), dtype=np.float32))

    assert all(block.shape[0] <= 4 for _, block in chunk_source(source, chunk_rows=4))


def test_an_empty_source_yields_no_chunks() -> None:
    assert list(chunk_source(_ArraySource(np.zeros((0, 2), dtype=np.float32)), 4)) == []


def test_a_zero_chunk_is_refused_rather_than_looping_forever() -> None:
    source = _ArraySource(np.zeros((4, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="at least 1"):
        list(chunk_source(source, chunk_rows=0))


# ------------------------------------------------- ground truth without the corpus


def test_only_the_neighbour_vectors_are_read() -> None:
    """The whole point: k x Q vectors instead of N. At a billion rows the
    difference is between a gigabyte and half a terabyte."""
    corpus = np.arange(100, dtype=np.float32).reshape(50, 2)
    source = _ArraySource(corpus)
    neighbour_ids = np.array([[0, 49], [7, 8]], dtype=np.int64)

    gathered, read_rows = neighbour_vectors(source, neighbour_ids)

    assert gathered.shape == (2, 2, 2)
    assert read_rows == 4
    np.testing.assert_array_equal(gathered[0, 0], corpus[0])
    np.testing.assert_array_equal(gathered[0, 1], corpus[49])
    np.testing.assert_array_equal(gathered[1, 0], corpus[7])


def test_a_neighbour_id_outside_the_corpus_is_refused() -> None:
    """A published dataset subsampled without remapping its neighbour ids points
    at rows that no longer exist, and silently dropping them would inflate recall."""
    source = _ArraySource(np.zeros((10, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="outside the corpus"):
        neighbour_vectors(source, np.array([[0, 10]], dtype=np.int64))


def test_duplicate_neighbour_ids_are_read_once() -> None:
    """Queries share neighbours. Reading each row once is what keeps the cost
    proportional to distinct neighbours rather than to k times Q."""
    source = _ArraySource(np.arange(20, dtype=np.float32).reshape(10, 2))
    ids = np.array([[0, 1], [1, 0]], dtype=np.int64)

    _, read_rows = neighbour_vectors(source, ids)

    assert read_rows == 2


# ------------------------------------------- the adapter streams from a source


def test_the_adapter_loads_from_a_source_without_holding_the_corpus() -> None:
    """`load_dataset_streaming` reads chunk by chunk, so the resident set is one
    chunk rather than the corpus. This is the difference between a scale that fits
    in RAM and one that does not."""
    from theodb_bench.adapters.base import VectorTableSpec
    from theodb_bench.adapters.postgres import PgvectorAdapter

    class _Server:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.written: list[bytes] = []
            self.reads: list[tuple[int, int]] = []

        def execute(self, sql: str, parameters: object = None) -> None:
            self.statements.append(sql)

        def fetch_one(self, sql: str, parameters: object = None):
            return (12,) if "count(*)" in sql else None

        def cursor(self):
            return _Cursor(self)

    class _Cursor:
        def __init__(self, server: _Server) -> None:
            self._server = server

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def copy(self, sql: str):
            self._server.statements.append(sql)
            return _Writer(self._server)

    class _Writer:
        def __init__(self, server: _Server) -> None:
            self._server = server

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def write(self, payload: bytes) -> None:
            self._server.written.append(payload)

    class _CountingSource(_ArraySource):
        def __init__(self, array: np.ndarray, log: list[tuple[int, int]]) -> None:
            super().__init__(array)
            self._log = log

        def rows(self, start: int, stop: int) -> np.ndarray:
            self._log.append((start, stop))
            return super().rows(start, stop)

    server = _Server()
    adapter = PgvectorAdapter()
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]
    adapter._cursor = server.cursor  # type: ignore[method-assign]

    reads: list[tuple[int, int]] = []
    source = _CountingSource(np.zeros((12, 4), dtype=np.float32), reads)

    outcome = adapter.load_dataset_streaming(
        VectorTableSpec(table="big", dimension=4, metric="l2"), source, chunk_rows=5
    )

    assert reads == [(0, 5), (5, 10), (10, 12)], "the corpus was not read in chunks"
    assert "FORMAT BINARY" in server.statements[-1] or any(
        "FORMAT BINARY" in s for s in server.statements
    )
    assert outcome.rows_expected == 12


# --------------------------------- ground truth over a corpus that is not resident
#
# `brute_force_ground_truth` chunks over queries and holds the corpus. At 20 000 000
# x 128 float32 that is 10.2 GB, on a host that also runs three PostgreSQL
# containers. The corpus has to be the thing that streams.


def test_streaming_ground_truth_matches_the_resident_version() -> None:
    """The equivalence that makes the streaming oracle usable at all."""
    from theodb_bench.analysis.quality import brute_force_ground_truth
    from theodb_bench.streaming import streaming_ground_truth

    rng = np.random.default_rng(20260817)
    corpus = rng.random((500, 12), dtype=np.float32)
    queries = rng.random((7, 12), dtype=np.float32)

    ids, dists = streaming_ground_truth(_ArraySource(corpus), queries, k=10, chunk_rows=64)
    ref_ids, ref_dists = brute_force_ground_truth(corpus, queries, 10)

    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_allclose(dists, ref_dists, rtol=1e-6, atol=0.0)


def test_the_chunk_size_does_not_change_the_answer() -> None:
    """A top-k maintained across chunks must not depend on where the chunks fell."""
    from theodb_bench.streaming import streaming_ground_truth

    rng = np.random.default_rng(7)
    corpus = rng.random((300, 8), dtype=np.float32)
    queries = rng.random((5, 8), dtype=np.float32)
    source = _ArraySource(corpus)

    first, _ = streaming_ground_truth(source, queries, k=5, chunk_rows=7)
    second, _ = streaming_ground_truth(source, queries, k=5, chunk_rows=256)

    np.testing.assert_array_equal(first, second)


def test_ties_across_a_chunk_boundary_still_resolve_by_id() -> None:
    """The case a running top-k gets wrong: equal distances split by a boundary.

    Every vector here is identical, so only ascending id can decide, and the
    boundary falls in the middle of the tie.
    """
    from theodb_bench.streaming import streaming_ground_truth

    corpus = np.ones((40, 3), dtype=np.float32)
    queries = np.zeros((2, 3), dtype=np.float32)

    ids, _ = streaming_ground_truth(_ArraySource(corpus), queries, k=6, chunk_rows=5)

    for row in ids:
        np.testing.assert_array_equal(row, np.arange(6))


def test_the_corpus_is_read_in_bounded_slices() -> None:
    from theodb_bench.streaming import streaming_ground_truth

    reads: list[tuple[int, int]] = []

    class _Counting(_ArraySource):
        def rows(self, start: int, stop: int) -> np.ndarray:
            reads.append((start, stop))
            return super().rows(start, stop)

    rng = np.random.default_rng(1)
    source = _Counting(rng.random((100, 4), dtype=np.float32))

    streaming_ground_truth(source, rng.random((3, 4), dtype=np.float32), k=5, chunk_rows=25)

    assert reads == [(0, 25), (25, 50), (50, 75), (75, 100)]
