"""Reading ANN benchmark dataset files.

Two formats cover the public ANN corpora: the HDF5 layout ANN-Benchmarks
publishes (``train``, ``test``, ``neighbors``, ``distances``) and the ``fvecs``
family used by SIFT and GIST.

One rule governs both readers, and it is the reason this module exists rather
than a two-line ``h5py`` call at the call site:

**Published distances are not read.** Every ANN file ships neighbour ids *and*
their distances. The distances were produced by someone else's precision and
metric convention, and using them would make our recall a comparison against a
number we cannot defend. The ids are used; the distances are recomputed from
the vectors (`MEASUREMENT-INTEGRITY.md` I3).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.errors import DatasetError, ErrorContext, Phase

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

HDF5_TRAIN: Final[str] = "train"
HDF5_TEST: Final[str] = "test"
HDF5_NEIGHBORS: Final[str] = "neighbors"


@dataclass(frozen=True)
class AnnDataset:
    """A loaded ANN corpus.

    ``neighbors`` holds published ids when the file provides them. Their
    distances are deliberately absent: recall recomputes them.
    """

    train: FloatArray
    test: FloatArray
    neighbors: IntArray | None
    source: Path
    metric: str | None = None

    @property
    def dimension(self) -> int:
        return int(self.train.shape[1])

    @property
    def corpus_size(self) -> int:
        return int(self.train.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.test.shape[0])

    def subsample(self, corpus_size: int, query_count: int) -> AnnDataset:
        """Take a prefix of the corpus and query set.

        A prefix, not a random sample: a sampled corpus would invalidate the
        published neighbour ids, which index into the full train array. When
        the corpus is reduced the neighbours are dropped rather than silently
        reinterpreted against different vectors.
        """
        if corpus_size > self.corpus_size or query_count > self.query_count:
            raise DatasetError(
                f"cannot take {corpus_size}x{query_count} from a "
                f"{self.corpus_size}x{self.query_count} dataset",
                context=ErrorContext(phase=Phase.DATASET_LOAD),
            )
        reduced = corpus_size < self.corpus_size
        return AnnDataset(
            train=self.train[:corpus_size],
            test=self.test[:query_count],
            # Published ids index the full corpus; against a prefix they would
            # point at the wrong vectors.
            neighbors=None
            if reduced
            else (self.neighbors[:query_count] if self.neighbors is not None else None),
            source=self.source,
            metric=self.metric,
        )


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise DatasetError(
            "reading an ANN HDF5 dataset needs h5py; "
            "install it with: pip install 'theodb-bench[datasets]'",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
            cause=exc,
        ) from exc
    return h5py


def read_ann_hdf5(path: Path) -> AnnDataset:
    """Read an ANN-Benchmarks HDF5 file.

    Reads ``train``, ``test`` and ``neighbors``. Does **not** read
    ``distances``: see the module docstring.
    """
    h5py = _require_h5py()
    if not path.is_file():
        raise DatasetError(
            f"no dataset file at {path}", context=ErrorContext(phase=Phase.DATASET_LOAD)
        )
    try:
        with h5py.File(path, "r") as handle:
            missing = [key for key in (HDF5_TRAIN, HDF5_TEST) if key not in handle]
            if missing:
                raise DatasetError(
                    f"{path} is not an ANN-Benchmarks file: missing {', '.join(missing)}",
                    context=ErrorContext(phase=Phase.DATASET_LOAD),
                )
            train = np.asarray(handle[HDF5_TRAIN][:], dtype=np.float32)
            test = np.asarray(handle[HDF5_TEST][:], dtype=np.float32)
            neighbors = (
                np.asarray(handle[HDF5_NEIGHBORS][:], dtype=np.int64)
                if HDF5_NEIGHBORS in handle
                else None
            )
            metric = handle.attrs.get("distance")
    except OSError as exc:
        raise DatasetError(
            f"{path} could not be read as HDF5",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
            cause=exc,
        ) from exc

    _check_shapes(path, train, test, neighbors)
    return AnnDataset(
        train=train,
        test=test,
        neighbors=neighbors,
        source=path,
        metric=str(metric) if metric is not None else None,
    )


def read_fvecs(path: Path, limit: int | None = None) -> FloatArray:
    """Read the ``fvecs`` format: per vector, an int32 dimension then floats.

    The dimension is repeated in every record. It is checked on every record
    rather than trusted from the first: a truncated or concatenated file
    otherwise parses into plausible garbage.
    """
    if not path.is_file():
        raise DatasetError(
            f"no dataset file at {path}", context=ErrorContext(phase=Phase.DATASET_LOAD)
        )
    vectors: list[FloatArray] = []
    dimension: int | None = None
    with path.open("rb") as handle:
        while True:
            header = handle.read(4)
            if not header:
                break
            if len(header) < 4:
                raise DatasetError(
                    f"{path} ends mid-header after {len(vectors)} vectors",
                    context=ErrorContext(phase=Phase.DATASET_LOAD),
                )
            (record_dimension,) = struct.unpack("<i", header)
            if record_dimension <= 0:
                raise DatasetError(
                    f"{path}: vector {len(vectors)} declares dimension {record_dimension}",
                    context=ErrorContext(phase=Phase.DATASET_LOAD),
                )
            if dimension is None:
                dimension = record_dimension
            elif record_dimension != dimension:
                raise DatasetError(
                    f"{path}: vector {len(vectors)} has dimension {record_dimension}, "
                    f"earlier vectors have {dimension}",
                    context=ErrorContext(phase=Phase.DATASET_LOAD),
                )
            payload = handle.read(4 * record_dimension)
            if len(payload) < 4 * record_dimension:
                raise DatasetError(
                    f"{path} is truncated inside vector {len(vectors)}",
                    context=ErrorContext(phase=Phase.DATASET_LOAD),
                )
            vectors.append(np.frombuffer(payload, dtype=np.float32))
            if limit is not None and len(vectors) >= limit:
                break

    if not vectors:
        raise DatasetError(
            f"{path} contains no vectors", context=ErrorContext(phase=Phase.DATASET_LOAD)
        )
    return np.vstack(vectors).astype(np.float32)


def read_ivecs(path: Path, limit: int | None = None) -> IntArray:
    """Read the ``ivecs`` format: the integer twin of fvecs, used for neighbours."""
    if not path.is_file():
        raise DatasetError(
            f"no dataset file at {path}", context=ErrorContext(phase=Phase.DATASET_LOAD)
        )
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise DatasetError(
            f"{path} contains no vectors", context=ErrorContext(phase=Phase.DATASET_LOAD)
        )
    dimension = int(raw[0])
    if dimension <= 0 or raw.size % (dimension + 1) != 0:
        raise DatasetError(
            f"{path} is not a valid ivecs file: dimension {dimension} does not divide "
            f"{raw.size} words",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    reshaped = raw.reshape(-1, dimension + 1)
    if not np.all(reshaped[:, 0] == dimension):
        raise DatasetError(
            f"{path}: not every record declares dimension {dimension}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    rows = reshaped[:limit] if limit is not None else reshaped
    return np.asarray(rows[:, 1:], dtype=np.int64)


def _check_shapes(
    path: Path, train: FloatArray, test: FloatArray, neighbors: IntArray | None
) -> None:
    if train.ndim != 2 or test.ndim != 2:
        raise DatasetError(
            f"{path}: train and test must be 2-D, got {train.shape} and {test.shape}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if train.shape[1] != test.shape[1]:
        raise DatasetError(
            f"{path}: train has dimension {train.shape[1]}, test has {test.shape[1]}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if neighbors is None:
        return
    if neighbors.shape[0] != test.shape[0]:
        raise DatasetError(
            f"{path}: {neighbors.shape[0]} neighbour rows for {test.shape[0]} queries",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if neighbors.size and (neighbors.min() < 0 or neighbors.max() >= train.shape[0]):
        # NumPy would wrap a negative index into a real vector and produce
        # confident ground truth for the wrong neighbour.
        raise DatasetError(
            f"{path}: neighbour ids fall outside the corpus of {train.shape[0]} vectors",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
