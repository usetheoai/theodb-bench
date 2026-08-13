"""Dataset readers. Published distances are never read; ids are."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.errors import DatasetError
from theodb_bench.formats import read_ann_hdf5, read_fvecs, read_ivecs

h5py = pytest.importorskip("h5py")


def _write_hdf5(
    path: Path,
    *,
    train: npt.NDArray[np.generic] | None = None,
    test: npt.NDArray[np.generic] | None = None,
    neighbors: npt.NDArray[np.generic] | None = None,
    distances: npt.NDArray[np.generic] | None = None,
    metric: str | None = "euclidean",
) -> Path:
    with h5py.File(path, "w") as handle:
        if train is not None:
            handle.create_dataset("train", data=train)
        if test is not None:
            handle.create_dataset("test", data=test)
        if neighbors is not None:
            handle.create_dataset("neighbors", data=neighbors)
        if distances is not None:
            handle.create_dataset("distances", data=distances)
        if metric is not None:
            handle.attrs["distance"] = metric
    return path


def _write_fvecs(path: Path, vectors: npt.NDArray[np.float32]) -> Path:
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack("<i", len(vector)))
            handle.write(np.asarray(vector, dtype=np.float32).tobytes())
    return path


# --------------------------------------------------------------------- HDF5


def test_reading_an_ann_file_gives_train_test_and_neighbours(tmp_path: Path) -> None:
    train = np.random.default_rng(1).standard_normal((50, 8)).astype(np.float32)
    test = train[:5]
    neighbors = np.tile(np.arange(3), (5, 1))
    dataset = read_ann_hdf5(
        _write_hdf5(tmp_path / "d.hdf5", train=train, test=test, neighbors=neighbors)
    )
    assert dataset.corpus_size == 50
    assert dataset.query_count == 5
    assert dataset.dimension == 8
    assert dataset.neighbors is not None and dataset.neighbors.shape == (5, 3)
    assert dataset.metric == "euclidean"


def test_published_distances_are_not_read(tmp_path: Path) -> None:
    # They carry someone else's precision and metric convention; recall
    # recomputes them from the vectors instead.
    train = np.zeros((10, 4), dtype=np.float32)
    path = _write_hdf5(
        tmp_path / "d.hdf5",
        train=train,
        test=train[:2],
        neighbors=np.zeros((2, 3), dtype=np.int64),
        distances=np.full((2, 3), 999.0, dtype=np.float32),
    )
    dataset = read_ann_hdf5(path)
    assert not hasattr(dataset, "distances")
    assert 999.0 not in np.asarray(dataset.train)


def test_a_file_without_train_is_refused(tmp_path: Path) -> None:
    path = _write_hdf5(tmp_path / "d.hdf5", test=np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(DatasetError, match="missing train"):
        read_ann_hdf5(path)


def test_mismatched_dimensions_are_refused(tmp_path: Path) -> None:
    path = _write_hdf5(
        tmp_path / "d.hdf5",
        train=np.zeros((10, 4), dtype=np.float32),
        test=np.zeros((2, 8), dtype=np.float32),
    )
    with pytest.raises(DatasetError, match="has dimension"):
        read_ann_hdf5(path)


def test_neighbour_rows_must_match_the_query_count(tmp_path: Path) -> None:
    path = _write_hdf5(
        tmp_path / "d.hdf5",
        train=np.zeros((10, 4), dtype=np.float32),
        test=np.zeros((5, 4), dtype=np.float32),
        neighbors=np.zeros((2, 3), dtype=np.int64),
    )
    with pytest.raises(DatasetError, match="neighbour rows"):
        read_ann_hdf5(path)


def test_out_of_range_neighbour_ids_are_refused(tmp_path: Path) -> None:
    # NumPy would wrap a negative index into a real vector and produce
    # confident ground truth for the wrong neighbour.
    path = _write_hdf5(
        tmp_path / "d.hdf5",
        train=np.zeros((10, 4), dtype=np.float32),
        test=np.zeros((2, 4), dtype=np.float32),
        neighbors=np.array([[0, 1, 99], [0, 1, 2]], dtype=np.int64),
    )
    with pytest.raises(DatasetError, match="outside the corpus"):
        read_ann_hdf5(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no dataset file"):
        read_ann_hdf5(tmp_path / "absent.hdf5")


def test_a_non_hdf5_file_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "not.hdf5"
    target.write_bytes(b"definitely not hdf5")
    with pytest.raises(DatasetError, match="could not be read as HDF5"):
        read_ann_hdf5(target)


# ----------------------------------------------------------------- subsample


def test_subsampling_takes_a_prefix(tmp_path: Path) -> None:
    train = np.arange(40, dtype=np.float32).reshape(10, 4)
    dataset = read_ann_hdf5(_write_hdf5(tmp_path / "d.hdf5", train=train, test=train[:4]))
    reduced = dataset.subsample(5, 2)
    assert reduced.corpus_size == 5
    assert reduced.query_count == 2
    assert np.array_equal(reduced.train, train[:5])


def test_reducing_the_corpus_drops_published_neighbours(tmp_path: Path) -> None:
    # Published ids index the full corpus; against a prefix they would point at
    # the wrong vectors.
    train = np.zeros((10, 4), dtype=np.float32)
    dataset = read_ann_hdf5(
        _write_hdf5(
            tmp_path / "d.hdf5",
            train=train,
            test=train[:4],
            neighbors=np.zeros((4, 3), dtype=np.int64),
        )
    )
    assert dataset.subsample(5, 2).neighbors is None
    # Keeping the whole corpus keeps them.
    assert dataset.subsample(10, 2).neighbors is not None


def test_subsampling_beyond_the_dataset_is_refused(tmp_path: Path) -> None:
    train = np.zeros((10, 4), dtype=np.float32)
    dataset = read_ann_hdf5(_write_hdf5(tmp_path / "d.hdf5", train=train, test=train[:2]))
    with pytest.raises(DatasetError, match="cannot take"):
        dataset.subsample(100, 2)


# ---------------------------------------------------------------------- vecs


def test_fvecs_round_trips(tmp_path: Path) -> None:
    vectors = np.random.default_rng(2).standard_normal((7, 5)).astype(np.float32)
    assert np.allclose(read_fvecs(_write_fvecs(tmp_path / "v.fvecs", vectors)), vectors)


def test_fvecs_honours_a_limit(tmp_path: Path) -> None:
    vectors = np.zeros((100, 3), dtype=np.float32)
    assert read_fvecs(_write_fvecs(tmp_path / "v.fvecs", vectors), limit=10).shape == (10, 3)


def test_a_truncated_fvecs_file_is_refused(tmp_path: Path) -> None:
    path = _write_fvecs(tmp_path / "v.fvecs", np.zeros((3, 4), dtype=np.float32))
    path.write_bytes(path.read_bytes()[:-6])
    with pytest.raises(DatasetError, match="truncated"):
        read_fvecs(path)


def test_an_inconsistent_dimension_is_refused(tmp_path: Path) -> None:
    # Checked on every record, not trusted from the first: a concatenated file
    # otherwise parses into plausible garbage.
    path = tmp_path / "v.fvecs"
    with path.open("wb") as handle:
        handle.write(struct.pack("<i", 4))
        handle.write(np.zeros(4, dtype=np.float32).tobytes())
        handle.write(struct.pack("<i", 8))
        handle.write(np.zeros(8, dtype=np.float32).tobytes())
    with pytest.raises(DatasetError, match="earlier vectors have"):
        read_fvecs(path)


def test_a_negative_dimension_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.fvecs"
    path.write_bytes(struct.pack("<i", -1) + b"\0" * 8)
    with pytest.raises(DatasetError, match="declares dimension -1"):
        read_fvecs(path)


def test_an_empty_fvecs_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.fvecs"
    path.write_bytes(b"")
    with pytest.raises(DatasetError, match="no vectors"):
        read_fvecs(path)


def test_ivecs_reads_neighbour_ids(tmp_path: Path) -> None:
    path = tmp_path / "n.ivecs"
    rows = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(struct.pack("<i", len(row)))
            handle.write(row.tobytes())
    assert np.array_equal(read_ivecs(path), rows.astype(np.int64))


def test_a_malformed_ivecs_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "n.ivecs"
    path.write_bytes(struct.pack("<i", 3) + np.array([0, 1], dtype=np.int32).tobytes())
    with pytest.raises(DatasetError, match="not a valid ivecs"):
        read_ivecs(path)
