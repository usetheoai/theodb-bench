"""Running external commands safely.

Adapters and collectors invoke external binaries. Every invocation here is
shell-free, argument-list based, time-bounded, and resolved through PATH by
name rather than by an interpolated string, so a dataset id or a benchmark
label can never become part of a command line.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0


@dataclass(frozen=True)
class CommandResult:
    """Outcome of an external command.

    ``timed_out`` is distinct from a non-zero exit: a command killed by the
    timeout produced no verdict at all, and treating that as failure would be
    an inference the runner is not entitled to make.
    """

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def which(binary: str) -> Path | None:
    """Locate an executable on PATH."""
    found = shutil.which(binary)
    return Path(found) if found is not None else None


def run_command(
    argv: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command and capture its output.

    Never raises for a failing command: probing the environment is expected to
    fail on hosts that lack a tool, and the caller decides whether that is an
    absence or an error.
    """
    args = tuple(argv)
    if not args:
        raise ValueError("run_command requires at least the program name")
    try:
        # S603: argv is a list and shell is never used, by construction of
        # this helper -- that is the whole reason it exists.
        completed = subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(args, None, "", "", timed_out=True)
    except (OSError, ValueError) as exc:
        return CommandResult(args, None, "", str(exc), timed_out=False)
    return CommandResult(
        args,
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
        timed_out=False,
    )


def first_line(text: str) -> str:
    """First non-empty line of command output, or the empty string."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
