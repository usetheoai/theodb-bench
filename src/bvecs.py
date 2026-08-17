"""Reading the BIGANN `bvecs` corpus as a streamable source.

The format, verified against the first bytes of the real file rather than taken
from documentation: each record is a little-endian int32 dimension followed by
that many `uint8` components. BIGANN's SIFT descriptors are 128-dimensional, so a
record is 132 bytes, and the first 20 000 000 of them are 2.64 GB.

Why this is a reader and not a `np.fromfile`. The corpus has to be addressable by
row range, because both the loader and the oracle stream — 20M x 128 float32 held
whole is 10.2 GB on a host that is also running the database under test. And the
dimension is repeated on every record, which makes a disagreement detectable:
reading the file flat would reshape past it and slice every later vector at the
wrong offset, producing a corpus that is subtly wrong everywhere rather than
loudly wrong once.

Rows come out as float32 because the column under test is `vector`, which is
float4. Handing the oracle `uint8` would let it compute in a precision the
database never sees, which is invariant I1 (`docs/methodology/`): the oracle sees
what the system sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

_DIMENSION_BYTES: Final[int] = 4


class BvecsSource:
    """A `bvecs` file read in row ranges, holding only the slice asked for."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        size = self._path.stat().st_size
        if size == 0:
            raise ValueError(f"{self._path} is empty: a corpus with no records is not a corpus")
        if size < _DIMENSION_BYTES:
            raise ValueError(
                f"{self._path} is {size} bytes, too short to hold even a record header"
            )

        with self._path.open("rb") as handle:
            self._dimension = int(np.frombuffer(handle.read(_DIMENSION_BYTES), dtype="<i4")[0])
        if self._dimension < 1:
            raise ValueError(
                f"{self._path} declares a dimension of {self._dimension} in its first record"
            )

        self._record_bytes = _DIMENSION_BYTES + self._dimension
        rows, remainder = divmod(size, self._record_bytes)
        if remainder:
            raise ValueError(
                f"{self._path} holds {size} bytes, which is not a whole number of records "
                f"of {self._record_bytes} bytes ({rows} records plus {remainder} bytes). A "
                f"fetch cut short mid-record would otherwise be read as a full vector."
            )
        self._row_count = int(rows)

        # One mapping for the life of the source: the pages the kernel keeps are
        # the ones recently touched, so the resident set follows the read pattern
        # rather than the file size.
        self._map = np.memmap(
            self._path, dtype=np.uint8, mode="r", shape=(self._row_count, self._record_bytes)
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def dimension(self) -> int:
        return self._dimension

    def rows(self, start: int, stop: int) -> npt.NDArray[np.floating[Any]]:
        """Rows `[start, stop)` as float32, checking the per-record dimension."""
        if start < 0 or stop > self._row_count or stop < start:
            raise ValueError(
                f"rows({start}, {stop}) is outside the corpus of {self._row_count} records"
            )
        if start == stop:
            return np.empty((0, self._dimension), dtype=np.float32)

        block = self._map[start:stop]
        declared = block[:, :_DIMENSION_BYTES].copy().view("<i4").reshape(-1)
        wrong = np.flatnonzero(declared != self._dimension)
        if wrong.size:
            row = int(start + wrong[0])
            raise ValueError(
                f"{self._path} record {row} declares dimension {int(declared[wrong[0]])} where "
                f"the first record declares {self._dimension}. The file is not a single corpus, "
                f"and reading past the disagreement would slice every later vector at the "
                f"wrong offset."
            )
        return block[:, _DIMENSION_BYTES:].astype(np.float32)
