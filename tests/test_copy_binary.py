"""The binary COPY stream, byte for byte.

Measured 2026-08-17 on one million SIFT-128 vectors: moving the bulk load from
batched INSERTs to a text COPY took it from 122 s to 75 s, and then **72 of those
75 seconds were `_to_column`** — a Python `repr()` per float, 128 million of them.
The round-trip reduction had already extracted everything it could; the remaining
cost is the text encoding itself.

At that rate a billion vectors is roughly twenty hours of load. Encoding the same
data in PostgreSQL's binary COPY format is a numpy byteswap and a `tobytes()`,
which is why the format is written out by hand here rather than left to a
per-value dumper.

Hand-writing a wire format is only safe if it is pinned. These tests assert the
exact bytes against the format PostgreSQL documents, and an integration test
loads them into a real server and reads the values back.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from theodb_bench.copy_binary import (
    BINARY_HEADER,
    BINARY_TRAILER,
    encode_vector_rows,
)


def test_the_stream_opens_with_the_signature_postgresql_requires() -> None:
    """`PGCOPY\\n\\377\\r\\n\\0`, then two int32 of zero: flags and header extension."""
    assert BINARY_HEADER == b"PGCOPY\n\xff\r\n\x00" + b"\x00" * 8


def test_the_stream_closes_with_the_end_of_data_marker() -> None:
    assert struct.pack(">h", -1) == BINARY_TRAILER


def test_a_row_carries_its_field_count_then_each_field_length_prefixed() -> None:
    vectors = np.array([[1.0, 2.0]], dtype=np.float32)

    body = encode_vector_rows(vectors, start_id=0)

    # int16 field count = 2
    assert body[0:2] == struct.pack(">h", 2)
    # field 1: int32 length 4, then the id as int32 big-endian
    assert body[2:6] == struct.pack(">i", 4)
    assert body[6:10] == struct.pack(">i", 0)
    # field 2: int32 length = 4 header bytes + 2 dims * 4 bytes
    assert body[10:14] == struct.pack(">i", 4 + 2 * 4)
    # pgvector's header: int16 dim, int16 unused
    assert body[14:18] == struct.pack(">hh", 2, 0)
    assert body[18:26] == struct.pack(">ff", 1.0, 2.0)


def test_ids_increment_from_the_supplied_start() -> None:
    vectors = np.zeros((3, 2), dtype=np.float32)

    body = encode_vector_rows(vectors, start_id=7)

    row_size = 2 + (4 + 4) + (4 + 4 + 2 * 4)
    ids = [struct.unpack(">i", body[i * row_size + 6 : i * row_size + 10])[0] for i in range(3)]
    assert ids == [7, 8, 9]


def test_float64_input_is_narrowed_to_float4_rather_than_written_wide() -> None:
    """A float8 written into a float4 field would corrupt every following byte."""
    vectors = np.array([[1.5, 2.5]], dtype=np.float64)

    body = encode_vector_rows(vectors, start_id=0)

    assert struct.unpack(">i", body[10:14])[0] == 4 + 2 * 4
    assert struct.unpack(">ff", body[18:26]) == (1.5, 2.5)


def test_the_encoding_round_trips_through_numpy() -> None:
    rng = np.random.default_rng(20260817)
    vectors = rng.random((64, 16), dtype=np.float32)

    body = encode_vector_rows(vectors, start_id=0)

    row_size = 2 + 8 + (4 + 4 + 16 * 4)
    assert len(body) == 64 * row_size
    for i in (0, 31, 63):
        offset = i * row_size + 18
        got = np.frombuffer(body[offset : offset + 16 * 4], dtype=">f4")
        np.testing.assert_array_equal(got.astype(np.float32), vectors[i])


def test_an_empty_corpus_encodes_to_nothing_rather_than_a_malformed_row() -> None:
    assert encode_vector_rows(np.zeros((0, 8), dtype=np.float32), start_id=0) == b""


def test_a_one_dimensional_probe_is_refused() -> None:
    """The encoder takes a corpus. A single vector passed here would be read as a
    corpus of scalars and produce a stream PostgreSQL accepts and misinterprets."""
    with pytest.raises(ValueError, match="2-D"):
        encode_vector_rows(np.zeros(8, dtype=np.float32), start_id=0)
