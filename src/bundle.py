"""The immutable run bundle.

A run bundle is the evidence. Once finalized, its raw measurements and its
manifest never change (TRD D3). Re-analysis may add new derived artifacts --
that is the whole point of separating orchestration from reporting -- but it
may never rewrite what was measured, because a number that can be rewritten
after the fact is not evidence.

Immutability is enforced two ways: a marker file records that the bundle is
closed, and the files themselves are made read-only. Neither stops a determined
`chmod`; both stop the accident this project actually has to prevent, which is
a re-analysis silently overwriting the measurement it disagrees with.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from theodb_bench.errors import ErrorContext, ImmutableBundleError, Phase
from theodb_bench.schemas import read_validated, write_validated

MANIFEST_SCHEMA_VERSION: Final[int] = 1
FINALIZED_MARKER: Final[str] = ".finalized"

RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9-]+-[a-z0-9_]+-[0-9a-f]{6,}$"
)

_SLUG_ALLOWED: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

# Artifacts that live at the bundle root, each validated against its schema.
ROOT_ARTIFACTS: Final[dict[str, str]] = {
    "manifest": "manifest.json",
    "environment": "environment.json",
    "benchmark": "benchmark.json",
    "system": "system.json",
    "dataset": "dataset.json",
    "validation": "validation.json",
    "result": "result.json",
}

DERIVED_ARTIFACTS: Final[dict[str, str]] = {
    "statistics": "statistics.json",
    "pareto": "pareto.json",
    "regression": "regression.json",
}


def slugify(value: str) -> str:
    """Reduce an identifier to the characters a run id may contain."""
    return _SLUG_ALLOWED.sub("-", value.lower()).strip("-")


def build_run_id(
    benchmark_id: str,
    system_id: str,
    *,
    now: datetime | None = None,
    entropy: str | None = None,
) -> str:
    """Compose a run id that is unique, sortable and self-describing.

    The trailing hash is derived from the identity of the run rather than from
    randomness, so re-deriving an id for the same run at the same instant gives
    the same answer.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed = entropy if entropy is not None else f"{benchmark_id}|{system_id}|{moment.isoformat()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    system_slug = _SLUG_ALLOWED.sub("_", system_id.lower()).strip("_")
    return f"{stamp}-{slugify(benchmark_id)}-{system_slug}-{digest}"


def _make_read_only(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        # A filesystem that refuses chmod does not invalidate the bundle; the
        # marker file still records that it is closed.
        pass


@dataclass
class RunBundle:
    """A directory holding one run's evidence."""

    root: Path
    run_id: str

    # ------------------------------------------------------------------ create

    @classmethod
    def create(
        cls,
        results_root: Path,
        *,
        benchmark_id: str,
        system_id: str,
        now: datetime | None = None,
        entropy: str | None = None,
    ) -> RunBundle:
        """Create a fresh bundle directory. Refuses to reuse an existing one."""
        run_id = build_run_id(benchmark_id, system_id, now=now, entropy=entropy)
        root = results_root / run_id
        if root.exists():
            raise ImmutableBundleError(
                f"run bundle {run_id} already exists at {root}",
                context=ErrorContext(phase=Phase.FINALIZATION, run_id=run_id),
            )
        for sub in ("raw", "derived", "report"):
            (root / sub).mkdir(parents=True)
        return cls(root=root, run_id=run_id)

    @classmethod
    def open(cls, root: Path) -> RunBundle:
        """Open an existing bundle for reading or re-analysis."""
        if not root.is_dir():
            raise ImmutableBundleError(
                f"no run bundle at {root}",
                context=ErrorContext(phase=Phase.OFFLINE),
            )
        return cls(root=root, run_id=root.name)

    # ------------------------------------------------------------------- state

    @property
    def finalized(self) -> bool:
        return (self.root / FINALIZED_MARKER).exists()

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    def _refuse_if_finalized(self, what: str) -> None:
        if self.finalized:
            raise ImmutableBundleError(
                f"run bundle {self.run_id} is finalized; {what} would rewrite evidence",
                context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id),
            )

    # --------------------------------------------------------------- artifacts

    def write_artifact(self, name: str, payload: Any) -> Path:
        """Write and validate a root artifact. Rejected once finalized."""
        if name not in ROOT_ARTIFACTS:
            raise ImmutableBundleError(
                f"{name!r} is not a bundle root artifact; known: "
                f"{', '.join(sorted(ROOT_ARTIFACTS))}",
                context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id),
            )
        self._refuse_if_finalized(f"writing {name}")
        path = self.root / ROOT_ARTIFACTS[name]
        write_validated(
            name, path, payload, context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id)
        )
        return path

    def read_artifact(self, name: str) -> Any:
        if name not in ROOT_ARTIFACTS and name not in DERIVED_ARTIFACTS:
            raise ImmutableBundleError(
                f"{name!r} is not a bundle artifact",
                context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id),
            )
        if name in ROOT_ARTIFACTS:
            path = self.root / ROOT_ARTIFACTS[name]
        else:
            path = self.derived_dir / DERIVED_ARTIFACTS[name]
        return read_validated(
            name, path, context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id)
        )

    def write_derived(self, name: str, payload: Any, *, overwrite: bool = False) -> Path:
        """Write a derived artifact.

        Permitted after finalization -- re-analysis is supposed to produce new
        derived output -- but never as an overwrite unless explicitly asked
        while the bundle is still open.
        """
        if name not in DERIVED_ARTIFACTS:
            raise ImmutableBundleError(
                f"{name!r} is not a derived artifact; "
                f"known: {', '.join(sorted(DERIVED_ARTIFACTS))}",
                context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id),
            )
        path = self.derived_dir / DERIVED_ARTIFACTS[name]
        if path.exists() and not overwrite:
            raise ImmutableBundleError(
                f"derived artifact {name} already exists in {self.run_id}; "
                "re-analysis must not overwrite an earlier derivation",
                context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id),
            )
        if path.exists() and overwrite:
            self._refuse_if_finalized(f"overwriting derived {name}")
            path.chmod(0o644)
        write_validated(
            name, path, payload, context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id)
        )
        return path

    # --------------------------------------------------------------------- raw

    def raw_path(self, filename: str) -> Path:
        """Path a collector may write to, rejecting traversal out of the bundle."""
        self._refuse_if_finalized(f"writing raw/{filename}")
        candidate = (self.raw_dir / filename).resolve()
        if not candidate.is_relative_to(self.raw_dir.resolve()):
            raise ImmutableBundleError(
                f"raw artifact path {filename!r} escapes the bundle",
                context=ErrorContext(phase=Phase.OFFLINE, run_id=self.run_id),
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def write_raw_text(self, filename: str, content: str) -> Path:
        path = self.raw_path(filename)
        path.write_text(content, encoding="utf-8")
        return path

    def append_raw_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        path = self.raw_path(filename)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ---------------------------------------------------------------- finalize

    def finalize(self, manifest: dict[str, Any]) -> Path:
        """Write the manifest, freeze the bundle, and record the closure.

        The manifest is the last thing written, so a bundle carrying one is a
        bundle whose measurement phase completed.
        """
        self._refuse_if_finalized("finalizing again")
        if manifest.get("run_id") != self.run_id:
            raise ImmutableBundleError(
                f"manifest run_id {manifest.get('run_id')!r} does not match bundle {self.run_id!r}",
                context=ErrorContext(phase=Phase.FINALIZATION, run_id=self.run_id),
            )
        path = self.write_artifact("manifest", manifest)
        marker = self.root / FINALIZED_MARKER
        marker.write_text(
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z") + "\n",
            encoding="utf-8",
        )
        self._freeze()
        return path

    def _freeze(self) -> None:
        """Make measured evidence read-only.

        `derived/` and `report/` stay writable: re-analysis is expected to add
        to them, and doing so does not touch a measurement.
        """
        for artifact in ROOT_ARTIFACTS.values():
            candidate = self.root / artifact
            if candidate.exists():
                _make_read_only(candidate)
        for dirpath, _, filenames in os.walk(self.raw_dir):
            for filename in filenames:
                _make_read_only(Path(dirpath) / filename)

    # ---------------------------------------------------------------- contents

    def artifacts(self) -> dict[str, Path]:
        """Every artifact present in this bundle, by logical name."""
        found: dict[str, Path] = {}
        for name, filename in ROOT_ARTIFACTS.items():
            candidate = self.root / filename
            if candidate.exists():
                found[name] = candidate
        for name, filename in DERIVED_ARTIFACTS.items():
            candidate = self.derived_dir / filename
            if candidate.exists():
                found[name] = candidate
        return found

    def raw_files(self) -> list[Path]:
        if not self.raw_dir.is_dir():
            return []
        return sorted(p for p in self.raw_dir.rglob("*") if p.is_file())
