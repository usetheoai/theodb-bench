"""Resource isolation and, more importantly, detecting when it did not hold.

Priority order is correctness, detectability, portability, explicit fallback
(objective section 12). The hardest requirement is the third one: no subprocess
of the system under test may silently escape the declared controls. "Silently"
is the operative word -- this module is built so that an unenforceable control
is reported as unenforced rather than assumed.

Applying a cgroup usually needs privileges this process does not have.
Detecting a violation does not: every process's allowed CPU set is readable
from procfs, so escapes are caught even when nothing could be enforced.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from theodb_bench.absent import Absent, Measured, encode, unavailable, unsupported
from theodb_bench.errors import ErrorContext, IsolationError, Phase, ProcessEscapedError

_PROC: Final[Path] = Path("/proc")
_CGROUP_ROOT: Final[Path] = Path("/sys/fs/cgroup")


# --------------------------------------------------------------------- cpu sets


def parse_cpu_set(spec: str) -> frozenset[int]:
    """Parse a Linux CPU list such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, _, end_text = piece.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise IsolationError(
                    f"malformed CPU range {piece!r} in {spec!r}",
                    context=ErrorContext(phase=Phase.ISOLATION),
                    cause=exc,
                ) from exc
            if end < start:
                raise IsolationError(
                    f"descending CPU range {piece!r} in {spec!r}",
                    context=ErrorContext(phase=Phase.ISOLATION),
                )
            cpus.update(range(start, end + 1))
        else:
            try:
                cpus.add(int(piece))
            except ValueError as exc:
                raise IsolationError(
                    f"malformed CPU id {piece!r} in {spec!r}",
                    context=ErrorContext(phase=Phase.ISOLATION),
                    cause=exc,
                ) from exc
    if not cpus:
        raise IsolationError(
            f"CPU set {spec!r} selects no CPUs",
            context=ErrorContext(phase=Phase.ISOLATION),
        )
    return frozenset(cpus)


def format_cpu_set(cpus: frozenset[int] | set[int]) -> str:
    """Render a CPU set back to the compact Linux notation."""
    if not cpus:
        return ""
    ordered = sorted(cpus)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append((start, previous))
        start = previous = cpu
    ranges.append((start, previous))
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


def online_cpus() -> frozenset[int]:
    """CPUs this process is allowed to run on."""
    if hasattr(os, "sched_getaffinity"):
        return frozenset(os.sched_getaffinity(0))
    count = os.cpu_count() or 1
    return frozenset(range(count))


# ------------------------------------------------------------------ procfs read


def process_exists(pid: int) -> bool:
    return (_PROC / str(pid)).is_dir()


def process_children(pid: int) -> frozenset[int]:
    """Direct children of a process, read from procfs."""
    children: set[int] = set()
    task_dir = _PROC / str(pid) / "task"
    if not task_dir.is_dir():
        return frozenset()
    for task in task_dir.iterdir():
        try:
            raw = (task / "children").read_text(encoding="utf-8")
        except OSError:
            continue
        children.update(int(token) for token in raw.split() if token.isdigit())
    return frozenset(children)


def process_tree(root_pid: int) -> frozenset[int]:
    """Every process descending from ``root_pid``, including it.

    Read breadth-first from procfs. A process that forks and exits between two
    reads can be missed; that is a property of the kernel interface, not
    something to paper over, and the sampling collector runs often enough that
    a long-lived escapee is caught.
    """
    seen: set[int] = set()
    frontier = [root_pid]
    while frontier:
        pid = frontier.pop()
        if pid in seen or not process_exists(pid):
            continue
        seen.add(pid)
        frontier.extend(process_children(pid))
    return frozenset(seen)


def process_affinity(pid: int) -> Measured[frozenset[int]]:
    """The CPU set a process is actually allowed to run on."""
    try:
        raw = (_PROC / str(pid) / "status").read_text(encoding="utf-8")
    except OSError:
        return unavailable(f"/proc/{pid}/status unreadable")
    for line in raw.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            value = line.partition(":")[2].strip()
            try:
                return parse_cpu_set(value)
            except IsolationError:
                return unavailable(f"unparseable Cpus_allowed_list {value!r}")
    return unavailable(f"Cpus_allowed_list absent from /proc/{pid}/status")


def process_command(pid: int) -> str:
    try:
        raw = (_PROC / str(pid) / "cmdline").read_bytes()
    except OSError:
        return "<unreadable>"
    parts = [chunk.decode("utf-8", "replace") for chunk in raw.split(b"\0") if chunk]
    return " ".join(parts) if parts else "<empty>"


# --------------------------------------------------------------------- escapes


@dataclass(frozen=True)
class Escape:
    """A process observed outside the declared CPU allocation."""

    pid: int
    command: str
    allowed: str
    declared: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "command": self.command,
            "allowed": self.allowed,
            "declared": self.declared,
        }


def find_escapes(root_pid: int, declared_cpus: frozenset[int]) -> list[Escape]:
    """Every process in the tree whose CPU allocation exceeds what was declared.

    A process whose affinity cannot be read is not reported as an escape --
    that would be an inference. It surfaces through ``unreadable_processes``.
    """
    escapes: list[Escape] = []
    for pid in sorted(process_tree(root_pid)):
        allowed = process_affinity(pid)
        if isinstance(allowed, Absent):
            continue
        if not allowed <= declared_cpus:
            escapes.append(
                Escape(
                    pid=pid,
                    command=process_command(pid),
                    allowed=format_cpu_set(allowed),
                    declared=format_cpu_set(declared_cpus),
                )
            )
    return escapes


def unreadable_processes(root_pid: int) -> list[int]:
    """Processes in the tree whose affinity could not be read at all."""
    return [
        pid for pid in sorted(process_tree(root_pid)) if isinstance(process_affinity(pid), Absent)
    ]


# ----------------------------------------------------------------- exclusivity


class ExclusiveBenchmarkLock:
    """Whole-host mutual exclusion for measurement.

    Two benchmarks sharing a host measure each other. The lock is advisory and
    file-based, which is enough because every participant is this program.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise IsolationError(
                f"another benchmark holds the exclusive lock at {self.path}",
                context=ErrorContext(phase=Phase.ISOLATION),
                cause=exc,
            ) from exc
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> ExclusiveBenchmarkLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


# --------------------------------------------------------------------- cgroups


@dataclass(frozen=True)
class CgroupSupport:
    """Whether cgroup v2 controls can actually be applied here."""

    mounted: bool
    writable: bool
    controllers: frozenset[str]
    detail: str

    @property
    def usable(self) -> bool:
        return self.mounted and self.writable


def probe_cgroup_support(root: Path | None = None) -> CgroupSupport:
    base = root if root is not None else _CGROUP_ROOT
    controllers_file = base / "cgroup.controllers"
    if not controllers_file.exists():
        return CgroupSupport(False, False, frozenset(), f"cgroup v2 not mounted at {base}")
    try:
        controllers = frozenset(controllers_file.read_text(encoding="utf-8").split())
    except OSError as exc:
        return CgroupSupport(True, False, frozenset(), f"cgroup.controllers unreadable: {exc}")
    writable = os.access(base, os.W_OK)
    detail = (
        "cgroup v2 usable"
        if writable
        else f"{base} is not writable by uid {os.getuid()}; limits cannot be enforced"
    )
    return CgroupSupport(True, writable, controllers, detail)


# ------------------------------------------------------------------------ plan


@dataclass(frozen=True)
class IsolationPlan:
    """What the benchmark declared it would enforce."""

    cpu_set: frozenset[int] | None = None
    client_cpu_set: frozenset[int] | None = None
    memory_bytes: int | None = None
    numa_node: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_set": format_cpu_set(self.cpu_set) if self.cpu_set else None,
            "client_cpu_set": format_cpu_set(self.client_cpu_set) if self.client_cpu_set else None,
            "memory_bytes": self.memory_bytes,
            "numa_node": self.numa_node,
        }


@dataclass
class AppliedIsolation:
    """What was actually enforced, and what could not be."""

    plan: IsolationPlan
    cpu_affinity_applied: Measured[bool] = False
    memory_limit_applied: Measured[bool] = False
    numa_applied: Measured[bool] = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "declared": self.plan.as_dict(),
            "cpu_affinity_applied": encode(self.cpu_affinity_applied),
            "memory_limit_applied": encode(self.memory_limit_applied),
            "numa_applied": encode(self.numa_applied),
            "notes": list(self.notes),
        }


def apply_cpu_affinity(pid: int, cpus: frozenset[int]) -> Measured[bool]:
    """Pin a process to a CPU set, reporting honestly when it cannot be done."""
    if not hasattr(os, "sched_setaffinity"):
        return unsupported("sched_setaffinity is not available on this platform")
    available = online_cpus()
    unavailable_cpus = cpus - available
    if unavailable_cpus:
        raise IsolationError(
            f"declared CPUs {format_cpu_set(unavailable_cpus)} are not available to this process "
            f"(available: {format_cpu_set(available)})",
            context=ErrorContext(phase=Phase.ISOLATION, details={"pid": pid}),
        )
    try:
        os.sched_setaffinity(pid, cpus)
    except OSError as exc:
        return unavailable(f"sched_setaffinity failed for pid {pid}: {exc}")
    return True


def read_effective_memory_limit(cgroup_path: Path | None = None) -> Measured[int]:
    """Le o limite de memoria do cgroup em que ESTE processo ja roda, em bytes.

    Existe porque `apply_isolation` aconselhava "run under an externally created cgroup instead" e
    nunca verificava se alguem havia feito isso — deixando `memory_limit_applied` sempre ausente e,
    com ele, os perfis `nightly` e `release` inalcancaveis por construcao.

    Aplicar o limite aqui exigiria privilegio e teria efeito colateral sobre o host; LER o que ja
    vale nao tem nenhum dos dois. `max` significa ausencia de limite e e reportado como ausencia:
    devolver um numero faria um cgroup irrestrito passar por restrito, que e pior que reprovar,
    porque a corrida pareceria isolada.
    """
    base = cgroup_path if cgroup_path is not None else _cgroup_of_self()
    if base is None:
        return unavailable("could not resolve the cgroup this process runs in")
    arquivo = base / "memory.max"
    if not arquivo.exists():
        return unavailable(f"{arquivo} does not exist; no cgroup v2 memory limit in effect")
    try:
        texto = arquivo.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return unavailable(f"{arquivo} unreadable: {exc}")
    if texto == "max":
        return unavailable(f"{arquivo} is 'max': the cgroup imposes no memory limit")
    try:
        return int(texto)
    except ValueError:
        return unavailable(f"{arquivo} holds {texto!r}, which is not a byte count")


def _cgroup_of_self() -> Path | None:
    """O diretorio cgroup v2 deste processo, derivado de /proc/self/cgroup."""
    try:
        linhas = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for linha in linhas:
        # cgroup v2 e sempre a linha `0::<caminho>`.
        if linha.startswith("0::"):
            return _CGROUP_ROOT / linha[3:].lstrip("/")
    return None


def apply_isolation(
    plan: IsolationPlan, pid: int | None = None, cgroup_path: Path | None = None
) -> AppliedIsolation:
    """Apply what can be applied here and record what could not.

    Never claims enforcement it did not achieve: an unenforceable control is
    recorded as an absence, which is what later makes the run INVALID for a
    profile that requires isolation.
    """
    target = pid if pid is not None else os.getpid()
    applied = AppliedIsolation(plan=plan)

    if plan.cpu_set is not None:
        applied.cpu_affinity_applied = apply_cpu_affinity(target, plan.cpu_set)
        if isinstance(applied.cpu_affinity_applied, Absent):
            applied.notes.append(str(applied.cpu_affinity_applied))
    else:
        applied.cpu_affinity_applied = unavailable("no CPU set declared")

    if plan.memory_bytes is not None:
        # O check se chama "Declared memory bound was respected". Um cgroup externo MAIS APERTADO
        # que o declarado respeita a declaracao — ele a cumpre com folga. Um mais frouxo nao, e
        # dizer o contrario seria afirmar isolamento que nao existe.
        efetivo = read_effective_memory_limit(cgroup_path)
        if isinstance(efetivo, int):
            if efetivo <= plan.memory_bytes:
                applied.memory_limit_applied = True
                applied.notes.append(
                    f"external cgroup limit of {efetivo} bytes respects the declared "
                    f"{plan.memory_bytes}"
                )
            else:
                applied.memory_limit_applied = unavailable(
                    f"the cgroup allows {efetivo} bytes, above the declared "
                    f"{plan.memory_bytes}; the declared bound is not enforced"
                )
                applied.notes.append(str(applied.memory_limit_applied))
        else:
            applied.memory_limit_applied = efetivo
            applied.notes.append(str(applied.memory_limit_applied))
    else:
        applied.memory_limit_applied = unavailable("no memory bound declared")

    if plan.numa_node is not None:
        applied.numa_applied = unavailable(
            "NUMA placement requires numactl or libnuma bindings; not applied"
        )
        applied.notes.append(str(applied.numa_applied))
    else:
        applied.numa_applied = unavailable("no NUMA placement declared")

    return applied


def assert_no_escapes(root_pid: int, declared_cpus: frozenset[int]) -> None:
    """Raise if any process in the tree exceeds the declared allocation."""
    escapes = find_escapes(root_pid, declared_cpus)
    if escapes:
        rendered = "; ".join(f"pid {e.pid} on {e.allowed} ({e.command})" for e in escapes)
        raise ProcessEscapedError(
            f"{len(escapes)} process(es) outside the declared CPU set "
            f"{format_cpu_set(declared_cpus)}: {rendered}",
            context=ErrorContext(
                phase=Phase.VALIDATION,
                details={"escapes": [e.as_dict() for e in escapes]},
            ),
        )
