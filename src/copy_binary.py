"""PostgreSQL's binary COPY stream, built with numpy instead of per-value calls.

Measured on 2026-08-17, loading one million SIFT-128 vectors: batched INSERTs took
122 s, a text COPY took 75 s, and **72 of those 75 seconds were the text
encoding** — one Python `repr()` per float, 128 million of them. Cutting
round-trips had already given everything it could.

At 75 s per million, a billion vectors is about twenty hours of load, which is the
difference between a scale being measurable and being theoretical. The same data
in binary is a byteswap and a `tobytes()`: numpy writes the whole corpus in one
pass with no per-value Python.

The format is written out here rather than delegated to a per-type dumper because
psycopg has no binary dumper for `vector`, and adding one would mean a dependency
that helps only the pgvector-family adapters. Hand-writing a wire format is only
safe if it is pinned, so every field below is asserted byte for byte in
`tests/test_copy_binary.py`, and an integration test loads the result into a real
server and reads the values back.

Layout, per PostgreSQL's COPY BINARY documentation:

    header   b"PGCOPY\\n\\377\\r\\n\\0" + int32 flags(0) + int32 extension length(0)
    row      int16 field count, then per field: int32 byte length + the bytes
    trailer  int16 -1

and pgvector's own `vector` binary representation, which is the second field here:

    int16 dimensions, int16 unused, then `dimensions` big-endian float4
"""

from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

#: Opening bytes of any binary COPY stream: the signature, then two int32 of zero
#: for the flags field and the header-extension length.
BINARY_HEADER: bytes = b"PGCOPY\n\xff\r\n\x00" + b"\x00" * 8

#: A field count of -1 marks end of data.
BINARY_TRAILER: bytes = struct.pack(">h", -1)


def encode_vector_rows(vectors: npt.NDArray[np.floating], start_id: int) -> bytes:
    """Encode `(id, vector)` rows as the body of a binary COPY stream.

    The body only: the caller writes the header once before the first chunk and
    the trailer once after the last, so a corpus larger than memory can be
    streamed in pieces.

    Every row has the same byte length because the dimension is fixed, which is
    what lets the whole block be assembled by numpy rather than row by row.
    """
    array = np.asarray(vectors)
    if array.ndim != 2:
        raise ValueError(
            f"encode_vector_rows takes a 2-D corpus, got shape {array.shape}. A single "
            f"vector passed here would be read as a corpus of scalars and produce a "
            f"stream PostgreSQL accepts and misinterprets."
        )
    rows, dimension = array.shape
    if rows == 0:
        return b""

    # float4 big-endian, narrowed from whatever came in. A float8 written into a
    # float4 field would shift every following byte.
    payload = np.ascontiguousarray(array, dtype=">f4")

    vector_field_bytes = 4 + dimension * 4
    row_bytes = 2 + (4 + 4) + (4 + vector_field_bytes)

    block = np.empty((rows, row_bytes), dtype=np.uint8)

    # The parts that are identical on every row, written once and broadcast.
    prefix = (
        struct.pack(">h", 2)  # two fields: id, vector
        + struct.pack(">i", 4)  # id is 4 bytes
    )
    block[:, : len(prefix)] = np.frombuffer(prefix, dtype=np.uint8)

    ids = np.arange(start_id, start_id + rows, dtype=">i4")
    block[:, 6:10] = ids.view(np.uint8).reshape(rows, 4)

    middle = (
        struct.pack(">i", vector_field_bytes)
        + struct.pack(">h", dimension)
        + struct.pack(">h", 0)  # pgvector's unused half-word
    )
    block[:, 10 : 10 + len(middle)] = np.frombuffer(middle, dtype=np.uint8)

    block[:, 18:] = payload.view(np.uint8).reshape(rows, dimension * 4)

    return block.tobytes()
