"""Dataset identity, acquisition and verification.

A dataset is its checksums, not its filename (TRD D5). Nothing here trusts a
name, a size, or the fact that a file exists where one was expected: every path
that leads to a measurement passes through ``verify``.

Manifests are JSON rather than the YAML sketched in the TRD. The reason is that
manifests are validated artifacts like every other machine-readable file in this
project, and JSON keeps them inside the same schema machinery without adding a
parser dependency. See docs/decisions/0002-json-dataset-manifests.md.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from theodb_bench.errors import (
    ChecksumMismatchError,
    DatasetError,
    ErrorContext,
    Phase,
)
from theodb_bench.schemas import read_validated, validate

DATASET_SCHEMA_VERSION: Final[int] = 1
MANIFEST_SUFFIX: Final[str] = ".json"
_READ_CHUNK: Final[int] = 1024 * 1024
_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"https", "http", "file"})

DEFAULT_MANIFEST_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "datasets/manifests"
DEFAULT_DATASET_ROOT: Final[Path] = Path(".datasets")


def sha256_file(path: Path) -> str:
    """Streaming checksum, so a 100 GB dataset does not need 100 GB of RAM."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetError(
            f"could not read {path} to compute its checksum",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
            cause=exc,
        ) from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetFile:
    """One file belonging to a dataset."""

    path: str
    sha256: str
    size_bytes: int | None = None
    url: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    """A dataset's content-based identity."""

    id: str
    version: str
    license_name: str
    redistributable: bool
    files: tuple[DatasetFile, ...]
    preprocess_version: int
    source_url: str | None = None
    description: str | None = None
    properties: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DatasetManifest:
        validate("dataset", payload, context=ErrorContext(phase=Phase.DATASET_LOAD))
        source = payload.get("source") or {}
        return cls(
            id=payload["id"],
            version=payload["version"],
            license_name=payload["license"]["name"],
            redistributable=payload["license"]["redistributable"],
            files=tuple(
                DatasetFile(
                    path=entry["path"],
                    sha256=entry["sha256"],
                    size_bytes=entry.get("size_bytes"),
                    url=entry.get("url"),
                    role=entry.get("role"),
                )
                for entry in payload["files"]
            ),
            preprocess_version=payload["preprocess"]["version"],
            source_url=source.get("url"),
            description=payload.get("description"),
            properties=payload.get("properties"),
        )

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        payload = read_validated("dataset", path, context=ErrorContext(phase=Phase.DATASET_LOAD))
        return cls.from_payload(payload)

    def directory(self, root: Path) -> Path:
        """Where this dataset's files live under a dataset root."""
        return root / self.id / self.version

    def resolve(self, root: Path, entry: DatasetFile) -> Path:
        """Resolve a manifest path inside the dataset directory.

        A manifest is data, and data is not trusted: a path that escapes the
        dataset directory is rejected rather than followed.
        """
        base = self.directory(root)
        candidate = (base / entry.path).resolve()
        if not candidate.is_relative_to(base.resolve()):
            raise DatasetError(
                f"dataset {self.id}: file path {entry.path!r} escapes the dataset directory",
                context=ErrorContext(phase=Phase.DATASET_LOAD, details={"dataset": self.id}),
            )
        return candidate

    def file_by_role(self, role: str) -> DatasetFile | None:
        for entry in self.files:
            if entry.role == role:
                return entry
        return None


# --------------------------------------------------------------------- verify


@dataclass(frozen=True)
class FileVerification:
    """The outcome of checking one file against the manifest."""

    path: str
    present: bool
    expected_sha256: str
    observed_sha256: str | None
    size_bytes: int | None

    @property
    def ok(self) -> bool:
        return self.present and self.observed_sha256 == self.expected_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "present": self.present,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "size_bytes": self.size_bytes,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class DatasetVerification:
    """Whether a dataset on disk is the dataset the manifest identifies."""

    dataset_id: str
    version: str
    files: tuple[FileVerification, ...]

    @property
    def ok(self) -> bool:
        return all(entry.ok for entry in self.files)

    @property
    def missing(self) -> tuple[FileVerification, ...]:
        return tuple(entry for entry in self.files if not entry.present)

    @property
    def corrupt(self) -> tuple[FileVerification, ...]:
        return tuple(entry for entry in self.files if entry.present and not entry.ok)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_id,
            "version": self.version,
            "ok": self.ok,
            "files": [entry.as_dict() for entry in self.files],
        }


def verify_dataset(manifest: DatasetManifest, root: Path) -> DatasetVerification:
    """Check every file against its recorded checksum.

    Never raises for a mismatch: the caller decides whether a missing dataset
    is an error (about to run) or a fact to report (``dataset verify``).
    """
    outcomes: list[FileVerification] = []
    for entry in manifest.files:
        path = manifest.resolve(root, entry)
        if not path.is_file():
            outcomes.append(FileVerification(entry.path, False, entry.sha256, None, None))
            continue
        outcomes.append(
            FileVerification(
                path=entry.path,
                present=True,
                expected_sha256=entry.sha256,
                observed_sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return DatasetVerification(manifest.id, manifest.version, tuple(outcomes))


def require_verified(manifest: DatasetManifest, root: Path) -> DatasetVerification:
    """Verify, and refuse to continue unless every file matches.

    This is what a run calls. Measuring over unverified bytes produces a number
    about a dataset nobody can identify.
    """
    verification = verify_dataset(manifest, root)
    if verification.ok:
        return verification
    missing = [entry.path for entry in verification.missing]
    corrupt = [entry.path for entry in verification.corrupt]
    if corrupt:
        raise ChecksumMismatchError(
            f"dataset {manifest.id}: checksum mismatch for {', '.join(corrupt)}",
            context=ErrorContext(
                phase=Phase.DATASET_LOAD,
                details={"dataset": manifest.id, "corrupt": corrupt},
            ),
        )
    raise DatasetError(
        f"dataset {manifest.id}: missing {', '.join(missing)}; fetch it first",
        context=ErrorContext(
            phase=Phase.DATASET_LOAD,
            details={"dataset": manifest.id, "missing": missing},
        ),
    )


# ---------------------------------------------------------------------- fetch


def _check_url(url: str, dataset_id: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise DatasetError(
            f"dataset {dataset_id}: refusing URL scheme {parsed.scheme!r}",
            context=ErrorContext(phase=Phase.DATASET_LOAD, details={"url": url}),
        )


def _download(url: str, destination: Path, dataset_id: str, timeout: float) -> None:
    """Download to a temporary file in the destination directory, then rename.

    Writing in place would leave a half-downloaded file that looks like a
    dataset; the rename is atomic within a filesystem.
    """
    _check_url(url, dataset_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".partial")
    temporary = Path(temporary_name)
    try:
        # The URL scheme is checked against an allow-list in _check_url above.
        with (
            os.fdopen(handle, "wb") as sink,
            urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310
        ):
            shutil.copyfileobj(response, sink, _READ_CHUNK)
        temporary.replace(destination)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetError(
            f"dataset {dataset_id}: download failed for {url}",
            context=ErrorContext(phase=Phase.DATASET_LOAD, details={"url": url}),
            cause=exc,
        ) from exc


def fetch_dataset(
    manifest: DatasetManifest,
    root: Path,
    *,
    timeout: float = 300.0,
    force: bool = False,
) -> DatasetVerification:
    """Acquire every file the manifest declares, then verify.

    A file already present and matching is left alone. A file present and
    mismatched is a hard error rather than something to silently re-download:
    it means the bytes on disk are not what the manifest identifies, and the
    operator should know before it is replaced.
    """
    verification = verify_dataset(manifest, root)
    if verification.corrupt and not force:
        corrupt = [entry.path for entry in verification.corrupt]
        raise ChecksumMismatchError(
            f"dataset {manifest.id}: {', '.join(corrupt)} present but does not match the "
            "manifest; re-fetch explicitly with force if this is intended",
            context=ErrorContext(
                phase=Phase.DATASET_LOAD,
                details={"dataset": manifest.id, "corrupt": corrupt},
            ),
        )

    for entry in manifest.files:
        path = manifest.resolve(root, entry)
        if path.is_file() and not force and sha256_file(path) == entry.sha256:
            continue
        url = entry.url or _derive_url(manifest, entry)
        if url is None:
            raise DatasetError(
                f"dataset {manifest.id}: no URL for {entry.path} and no source to derive one from",
                context=ErrorContext(phase=Phase.DATASET_LOAD, details={"dataset": manifest.id}),
            )
        _download(url, path, manifest.id, timeout)
        observed = sha256_file(path)
        if observed != entry.sha256:
            path.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"dataset {manifest.id}: {entry.path} downloaded from {url} has sha256 "
                f"{observed}, manifest declares {entry.sha256}",
                context=ErrorContext(
                    phase=Phase.DATASET_LOAD,
                    details={"dataset": manifest.id, "file": entry.path, "url": url},
                ),
            )

    return require_verified(manifest, root)


def _derive_url(manifest: DatasetManifest, entry: DatasetFile) -> str | None:
    if manifest.source_url is None:
        return None
    base = manifest.source_url if manifest.source_url.endswith("/") else manifest.source_url + "/"
    return urllib.parse.urljoin(base, entry.path)


# -------------------------------------------------------------------- registry


@dataclass(frozen=True)
class DatasetRegistry:
    """The manifests this installation knows about."""

    manifest_dir: Path

    def paths(self) -> Iterator[Path]:
        if not self.manifest_dir.is_dir():
            return
        yield from sorted(self.manifest_dir.glob(f"*{MANIFEST_SUFFIX}"))

    def ids(self) -> list[str]:
        return [path.stem for path in self.paths()]

    def load(self, dataset_id: str) -> DatasetManifest:
        if not dataset_id or "/" in dataset_id or dataset_id.startswith("."):
            raise DatasetError(
                f"invalid dataset id {dataset_id!r}",
                context=ErrorContext(phase=Phase.DATASET_LOAD),
            )
        path = self.manifest_dir / f"{dataset_id}{MANIFEST_SUFFIX}"
        if not path.is_file():
            known = ", ".join(self.ids()) or "none installed"
            raise DatasetError(
                f"unknown dataset {dataset_id!r}; known datasets: {known}",
                context=ErrorContext(phase=Phase.DATASET_LOAD),
            )
        return DatasetManifest.load(path)

    def all(self) -> list[DatasetManifest]:
        return [DatasetManifest.load(path) for path in self.paths()]


def default_registry() -> DatasetRegistry:
    return DatasetRegistry(DEFAULT_MANIFEST_DIR)
