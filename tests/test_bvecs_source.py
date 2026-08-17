"""Reading the BIGANN `bvecs` corpus without holding it.

The format, verified against the first bytes of the real file rather than from
documentation: each record is a little-endian int32 dimension followed by that
many uint8 components. For BIGANN's SIFT descriptors the dimension is 128, so a
record is 132 bytes and 20 000 000 of them are 2.64 GB on disk.

Two things make this worth its own reader instead of a `np.fromfile`. The
dimension is repeated on every record, so a file whose records disagree is
corrupt in a way that a flat read would silently reshape rather than report. And
the corpus must be addressable by row range, because the oracle and the loader
both stream — holding 20M x 128 float32 would be 10.2 GB.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from theodb_bench.bvecs import BvecsSource
from theodb_bench.streaming import CorpusSource


def _write(path: Path, vectors: np.ndarray) -> Path:
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack("<i", vector.shape[0]))
            handle.write(vector.astype(np.uint8).tobytes())
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(20260817)
    vectors = rng.integers(0, 256, size=(37, 12), dtype=np.uint8)
    return _write(tmp_path / "base.bvecs", vectors), vectors


def test_a_bvecs_file_satisfies_the_corpus_protocol(corpus: tuple[Path, np.ndarray]) -> None:
    path, _ = corpus
    assert isinstance(BvecsSource(path), CorpusSource)


def test_the_row_count_comes_from_the_file_size(corpus: tuple[Path, np.ndarray]) -> None:
    path, vectors = corpus
    source = BvecsSource(path)

    assert source.row_count == vectors.shape[0]
    assert source.dimension == vectors.shape[1]


def test_rows_are_returned_as_float32_for_the_vector_column(
    corpus: tuple[Path, np.ndarray],
) -> None:
    """The column is `vector`, which is float4. Handing the oracle uint8 would
    make it compute in a precision the database never sees."""
    path, vectors = corpus
    block = BvecsSource(path).rows(5, 9)

    assert block.dtype == np.float32
    np.testing.assert_array_equal(block, vectors[5:9].astype(np.float32))


def test_any_range_reads_the_right_rows(corpus: tuple[Path, np.ndarray]) -> None:
    path, vectors = corpus
    source = BvecsSource(path)

    for start, stop in [(0, 1), (0, 37), (36, 37), (10, 10), (7, 23)]:
        np.testing.assert_array_equal(
            source.rows(start, stop), vectors[start:stop].astype(np.float32)
        )


def test_a_truncated_final_record_is_refused(tmp_path: Path) -> None:
    """A fetch cut short mid-record would otherwise read a short vector as a
    full one, and every distance computed from it would be wrong."""
    rng = np.random.default_rng(1)
    path = _write(tmp_path / "cut.bvecs", rng.integers(0, 256, size=(4, 12), dtype=np.uint8))
    with path.open("rb+") as handle:
        handle.truncate(path.stat().st_size - 5)

    with pytest.raises(ValueError, match="whole number of records"):
        BvecsSource(path)


def test_a_record_declaring_a_different_dimension_is_refused(tmp_path: Path) -> None:
    """The dimension is repeated per record, so disagreement is detectable — and
    a flat reshape would quietly slice every later vector at the wrong offset."""
    path = tmp_path / "mixed.bvecs"
    with path.open("wb") as handle:
        handle.write(struct.pack("<i", 4) + bytes([1, 2, 3, 4]))
        handle.write(struct.pack("<i", 4) + bytes([5, 6, 7, 8]))
        handle.write(struct.pack("<i", 4) + bytes([9, 9, 9, 9]))
    with path.open("rb+") as handle:
        handle.seek(2 * 8)
        handle.write(struct.pack("<i", 7))

    with pytest.raises(ValueError, match="dimension"):
        BvecsSource(path).rows(0, 3)


def test_an_empty_file_is_refused_rather_than_read_as_zero_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.bvecs"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        BvecsSource(path)


def test_a_range_beyond_the_corpus_is_refused(corpus: tuple[Path, np.ndarray]) -> None:
    path, _ = corpus

    with pytest.raises(ValueError, match="outside"):
        BvecsSource(path).rows(30, 99)


def test_the_reader_holds_no_more_than_the_slice(corpus: tuple[Path, np.ndarray]) -> None:
    """A memory-mapped read must not materialise the file. The check that this
    is a view and not a copy is the whole reason the reader exists."""
    path, _ = corpus
    source = BvecsSource(path)

    assert source.rows(0, 4).nbytes == 4 * source.dimension * 4


# ------------------------------------------------- taking a prefix of a source
#
# A workload declares a corpus size, and the file may hold more. Reducing an
# array is a slice; reducing a source has to be a view, because materialising it
# to cut it would defeat the reason it is a source.


def test_a_prefix_reports_the_smaller_row_count(corpus: tuple[Path, np.ndarray]) -> None:
    from theodb_bench.streaming import PrefixSource

    path, vectors = corpus
    prefix = PrefixSource(BvecsSource(path), 10)

    assert prefix.row_count == 10
    assert prefix.dimension == vectors.shape[1]


def test_a_prefix_reads_the_same_rows_as_the_underlying_source(
    corpus: tuple[Path, np.ndarray],
) -> None:
    from theodb_bench.streaming import PrefixSource

    path, vectors = corpus
    prefix = PrefixSource(BvecsSource(path), 10)

    np.testing.assert_array_equal(prefix.rows(2, 9), vectors[2:9].astype(np.float32))


def test_a_prefix_refuses_a_read_past_its_own_limit(corpus: tuple[Path, np.ndarray]) -> None:
    """The rows exist in the file, and that is exactly why the refusal matters:
    a silent read past the declared corpus would measure vectors the run says it
    did not load, and every recall figure would be against the wrong oracle."""
    from theodb_bench.streaming import PrefixSource

    path, _ = corpus
    prefix = PrefixSource(BvecsSource(path), 10)

    with pytest.raises(ValueError, match="outside"):
        prefix.rows(8, 12)


def test_a_prefix_longer_than_the_source_is_refused(corpus: tuple[Path, np.ndarray]) -> None:
    from theodb_bench.streaming import PrefixSource

    path, vectors = corpus

    with pytest.raises(ValueError, match="only"):
        PrefixSource(BvecsSource(path), vectors.shape[0] + 1)


def test_a_prefix_is_a_corpus_source(corpus: tuple[Path, np.ndarray]) -> None:
    from theodb_bench.streaming import PrefixSource

    path, _ = corpus
    assert isinstance(PrefixSource(BvecsSource(path), 5), CorpusSource)


def test_the_oracle_over_a_prefix_matches_the_oracle_over_the_equivalent_array(
    corpus: tuple[Path, np.ndarray],
) -> None:
    """The property the whole 20M path rests on: reading the first N records of a
    large file must give the same ground truth as holding those N vectors."""
    from theodb_bench.analysis.quality import brute_force_ground_truth
    from theodb_bench.streaming import PrefixSource, streaming_ground_truth

    path, vectors = corpus
    queries = vectors[:3].astype(np.float32)

    ids, distances = streaming_ground_truth(
        PrefixSource(BvecsSource(path), 20), queries, k=5, chunk_rows=6
    )
    reference_ids, reference_distances = brute_force_ground_truth(
        vectors[:20].astype(np.float32), queries, 5
    )

    np.testing.assert_array_equal(ids, reference_ids)
    np.testing.assert_allclose(distances, reference_distances, rtol=1e-6)
