"""A dataset is its checksums. Nothing here may trust a filename."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from theodb_bench.datasets import (
    DatasetManifest,
    DatasetRegistry,
    fetch_dataset,
    require_verified,
    sha256_file,
    verify_dataset,
)
from theodb_bench.errors import ChecksumMismatchError, DatasetError, SchemaValidationError


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_payload(files: list[dict[str, Any]], dataset_id: str = "toy") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": dataset_id,
        "version": "1",
        "license": {"name": "CC0-1.0", "redistributable": True},
        "files": files,
        "preprocess": {"version": 1},
    }


def _write_dataset(root: Path, dataset_id: str, contents: dict[str, bytes]) -> DatasetManifest:
    directory = root / dataset_id / "1"
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, payload in contents.items():
        (directory / name).write_bytes(payload)
        files.append({"path": name, "sha256": _digest(payload), "size_bytes": len(payload)})
    return DatasetManifest.from_payload(_manifest_payload(files, dataset_id))


# ------------------------------------------------------------------ checksums


def test_checksum_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(b"vectors")
    assert sha256_file(target) == _digest(b"vectors")


def test_checksum_of_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="could not read"):
        sha256_file(tmp_path / "absent.bin")


def test_checksum_streams_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    payload = b"x" * (3 * 1024 * 1024 + 17)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == _digest(payload)


# ------------------------------------------------------------------- manifest


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    payload = _manifest_payload([{"path": "a.bin", "sha256": _digest(b"a")}])
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = DatasetManifest.load(path)
    assert manifest.id == "toy"
    assert manifest.files[0].sha256 == _digest(b"a")


def test_manifest_without_a_checksum_is_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        DatasetManifest.from_payload(_manifest_payload([{"path": "a.bin"}]))


def test_manifest_path_may_not_escape_the_dataset_directory(tmp_path: Path) -> None:
    manifest = DatasetManifest.from_payload(
        _manifest_payload([{"path": "nested/../../escape.bin", "sha256": _digest(b"a")}])
    )
    with pytest.raises(DatasetError, match="escapes the dataset directory"):
        manifest.resolve(tmp_path, manifest.files[0])


def test_files_can_be_found_by_role() -> None:
    manifest = DatasetManifest.from_payload(
        _manifest_payload(
            [
                {"path": "base.fvecs", "sha256": _digest(b"a"), "role": "train"},
                {"path": "query.fvecs", "sha256": _digest(b"b"), "role": "queries"},
            ]
        )
    )
    found = manifest.file_by_role("queries")
    assert found is not None and found.path == "query.fvecs"
    assert manifest.file_by_role("neighbors") is None


# --------------------------------------------------------------------- verify


def test_a_matching_dataset_verifies(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha", "b.bin": b"beta"})
    verification = verify_dataset(manifest, tmp_path)
    assert verification.ok
    assert not verification.missing
    assert not verification.corrupt


def test_a_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha"})
    (tmp_path / "toy" / "1" / "a.bin").unlink()
    verification = verify_dataset(manifest, tmp_path)
    assert not verification.ok
    assert [entry.path for entry in verification.missing] == ["a.bin"]


def test_a_modified_file_is_detected(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha"})
    (tmp_path / "toy" / "1" / "a.bin").write_bytes(b"tampered")
    verification = verify_dataset(manifest, tmp_path)
    assert [entry.path for entry in verification.corrupt] == ["a.bin"]


def test_a_file_of_the_right_size_but_wrong_content_is_detected(tmp_path: Path) -> None:
    # The point of checksums: same name, same length, different bytes.
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha"})
    (tmp_path / "toy" / "1" / "a.bin").write_bytes(b"alpsa")
    assert not verify_dataset(manifest, tmp_path).ok


def test_require_verified_raises_on_corruption(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha"})
    (tmp_path / "toy" / "1" / "a.bin").write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        require_verified(manifest, tmp_path)


def test_require_verified_distinguishes_missing_from_corrupt(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha"})
    (tmp_path / "toy" / "1" / "a.bin").unlink()
    with pytest.raises(DatasetError, match="fetch it first") as excinfo:
        require_verified(manifest, tmp_path)
    assert not isinstance(excinfo.value, ChecksumMismatchError)


def test_verification_serialises_for_the_bundle(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path, "toy", {"a.bin": b"alpha"})
    payload = verify_dataset(manifest, tmp_path).as_dict()
    assert payload["ok"] is True
    assert payload["files"][0]["observed_sha256"] == payload["files"][0]["expected_sha256"]


# ---------------------------------------------------------------------- fetch


def test_fetch_downloads_and_verifies_from_a_file_url(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"vector payload"
    (source / "a.bin").write_bytes(payload)

    manifest = DatasetManifest.from_payload(
        _manifest_payload(
            [
                {
                    "path": "a.bin",
                    "sha256": _digest(payload),
                    "url": (source / "a.bin").as_uri(),
                }
            ]
        )
    )
    root = tmp_path / "datasets"
    verification = fetch_dataset(manifest, root)
    assert verification.ok
    assert (root / "toy" / "1" / "a.bin").read_bytes() == payload


def test_fetch_rejects_a_download_whose_checksum_disagrees(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.bin").write_bytes(b"actual content")

    manifest = DatasetManifest.from_payload(
        _manifest_payload(
            [
                {
                    "path": "a.bin",
                    "sha256": _digest(b"expected content"),
                    "url": (source / "a.bin").as_uri(),
                }
            ]
        )
    )
    root = tmp_path / "datasets"
    with pytest.raises(ChecksumMismatchError, match="downloaded from"):
        fetch_dataset(manifest, root)
    # The bad bytes must not be left behind looking like a dataset.
    assert not (root / "toy" / "1" / "a.bin").exists()


def test_fetch_leaves_an_already_correct_file_alone(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    manifest = _write_dataset(root, "toy", {"a.bin": b"alpha"})
    target = root / "toy" / "1" / "a.bin"
    before = target.stat().st_mtime_ns
    fetch_dataset(manifest, root)
    assert target.stat().st_mtime_ns == before


def test_fetch_refuses_to_silently_replace_a_mismatched_file(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    manifest = _write_dataset(root, "toy", {"a.bin": b"alpha"})
    (root / "toy" / "1" / "a.bin").write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError, match="re-fetch explicitly"):
        fetch_dataset(manifest, root)


def test_fetch_rejects_an_unsupported_url_scheme(tmp_path: Path) -> None:
    manifest = DatasetManifest.from_payload(
        _manifest_payload(
            [{"path": "a.bin", "sha256": _digest(b"a"), "url": "ftp://example.invalid/a.bin"}]
        )
    )
    with pytest.raises(DatasetError, match="refusing URL scheme"):
        fetch_dataset(manifest, tmp_path / "datasets")


def test_fetch_without_a_url_or_source_says_so(tmp_path: Path) -> None:
    manifest = DatasetManifest.from_payload(
        _manifest_payload([{"path": "a.bin", "sha256": _digest(b"a")}])
    )
    with pytest.raises(DatasetError, match="no URL"):
        fetch_dataset(manifest, tmp_path / "datasets")


# ------------------------------------------------------------------- registry


def test_registry_lists_and_loads_manifests(tmp_path: Path) -> None:
    payload = _manifest_payload([{"path": "a.bin", "sha256": _digest(b"a")}])
    (tmp_path / "toy.json").write_text(json.dumps(payload), encoding="utf-8")
    registry = DatasetRegistry(tmp_path)
    assert registry.ids() == ["toy"]
    assert registry.load("toy").id == "toy"
    assert len(registry.all()) == 1


def test_registry_reports_an_unknown_dataset_with_what_it_has(tmp_path: Path) -> None:
    (tmp_path / "toy.json").write_text(
        json.dumps(_manifest_payload([{"path": "a.bin", "sha256": _digest(b"a")}])),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="known datasets: toy"):
        DatasetRegistry(tmp_path).load("sift1m")


@pytest.mark.parametrize("dataset_id", ["", "../etc/passwd", "a/b", ".hidden"])
def test_registry_rejects_a_traversing_dataset_id(tmp_path: Path, dataset_id: str) -> None:
    with pytest.raises(DatasetError):
        DatasetRegistry(tmp_path).load(dataset_id)


def test_registry_on_a_missing_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert DatasetRegistry(tmp_path / "nowhere").ids() == []
