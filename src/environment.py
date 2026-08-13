"""Capturing the environment a run executed in.

This is what makes two runs comparable (TRD section 10). Anything that cannot
be determined is recorded as an explicit absence -- never as zero, never as an
empty string, and never guessed from a plausible default.

Linux is the supported measurement platform; on other systems most fields
resolve to ``unsupported`` rather than to a fabricated value.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from theodb_bench import __version__
from theodb_bench.absent import Absent, Measured, encode, unavailable, unsupported
from theodb_bench.command import first_line, run_command, which

ENVIRONMENT_SCHEMA_VERSION: Final[int] = 1

_PROC = Path("/proc")
_SYS = Path("/sys")


def _read_text(path: Path) -> str | None:
    """Read a small pseudo-file, returning None when it is not readable."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _linux_only(what: str) -> Absent:
    return unsupported(f"{what} is only captured on Linux; running on {sys.platform}")


# --------------------------------------------------------------------------- CPU


def _cpuinfo_blocks() -> list[dict[str, str]]:
    raw = _read_text(_PROC / "cpuinfo")
    if raw is None:
        return []
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        key, _, value = line.partition(":")
        current[key.strip()] = value.strip()
    if current:
        blocks.append(current)
    return blocks


def _cpu_topology(
    blocks: list[dict[str, str]],
) -> tuple[Measured[int], Measured[int], Measured[int]]:
    """Return (sockets, physical_cores, logical_cpus)."""
    if not blocks:
        missing = unavailable("/proc/cpuinfo unreadable")
        return missing, missing, missing
    logical = len(blocks)
    sockets: Measured[int]
    physical: Measured[int]
    socket_ids = {b["physical id"] for b in blocks if "physical id" in b}
    core_keys = {(b.get("physical id", "0"), b["core id"]) for b in blocks if "core id" in b}
    sockets = len(socket_ids) if socket_ids else unavailable("no physical id in /proc/cpuinfo")
    physical = len(core_keys) if core_keys else unavailable("no core id in /proc/cpuinfo")
    return sockets, physical, logical


def _cpu_cache() -> Measured[dict[str, str]]:
    base = _SYS / "devices/system/cpu/cpu0/cache"
    if not base.is_dir():
        return unavailable("sysfs cache hierarchy not exposed")
    levels: dict[str, str] = {}
    for index in sorted(base.glob("index*")):
        level = _read_text(index / "level")
        cache_type = _read_text(index / "type")
        size = _read_text(index / "size")
        if level is None or cache_type is None or size is None:
            continue
        levels[f"L{level}{cache_type[0].lower()}"] = size
    return levels if levels else unavailable("cache hierarchy present but unreadable")


def capture_cpu() -> dict[str, Any]:
    if not _is_linux():
        absent = _linux_only("CPU topology")
        return {
            key: encode(absent)
            for key in (
                "vendor",
                "model",
                "microarchitecture",
                "sockets",
                "physical_cores",
                "logical_cpus",
                "smt_enabled",
                "frequency_policy",
                "numa_nodes",
                "cache",
            )
        }

    blocks = _cpuinfo_blocks()
    sockets, physical, logical = _cpu_topology(blocks)
    first = blocks[0] if blocks else {}

    smt: Measured[bool]
    smt_control = _read_text(_SYS / "devices/system/cpu/smt/active")
    if smt_control in {"0", "1"}:
        smt = smt_control == "1"
    elif isinstance(physical, int) and isinstance(logical, int) and physical > 0:
        smt = logical > physical
    else:
        smt = unavailable("neither sysfs smt/active nor a usable core count")

    numa_root = _SYS / "devices/system/node"
    numa_nodes: Measured[int] = (
        len(list(numa_root.glob("node[0-9]*")))
        if numa_root.is_dir()
        else unavailable("sysfs NUMA topology not exposed")
    )

    governor = _read_text(_SYS / "devices/system/cpu/cpu0/cpufreq/scaling_governor")

    return {
        "vendor": encode(first.get("vendor_id") or unavailable("vendor_id absent")),
        "model": encode(first.get("model name") or unavailable("model name absent")),
        # Deliberately not derived: mapping family/model numbers to a
        # microarchitecture name requires a table we would have to keep correct.
        "microarchitecture": encode(unavailable("not derivable from /proc/cpuinfo")),
        "sockets": encode(sockets),
        "physical_cores": encode(physical),
        "logical_cpus": encode(logical),
        "smt_enabled": encode(smt),
        "frequency_policy": encode(governor or unavailable("cpufreq governor not exposed")),
        "numa_nodes": encode(numa_nodes),
        "cache": encode(_cpu_cache()),
    }


# ------------------------------------------------------------------------ memory


def _meminfo() -> dict[str, int]:
    raw = _read_text(_PROC / "meminfo")
    if raw is None:
        return {}
    values: dict[str, int] = {}
    for line in raw.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            amount = int(parts[0])
        except ValueError:
            continue
        # meminfo reports kB for sized entries and a bare count otherwise.
        values[key.strip()] = amount * 1024 if len(parts) > 1 and parts[1] == "kB" else amount
    return values


def _numa_memory() -> Measured[dict[str, int]]:
    root = _SYS / "devices/system/node"
    if not root.is_dir():
        return unavailable("sysfs NUMA topology not exposed")
    per_node: dict[str, int] = {}
    for node in sorted(root.glob("node[0-9]*")):
        raw = _read_text(node / "meminfo")
        if raw is None:
            continue
        for line in raw.splitlines():
            if "MemTotal" in line:
                parts = line.split()
                try:
                    per_node[node.name] = int(parts[-2]) * 1024
                except (ValueError, IndexError):
                    continue
                break
    return per_node if per_node else unavailable("per-node meminfo unreadable")


def capture_memory() -> dict[str, Any]:
    if not _is_linux():
        absent = _linux_only("memory topology")
        return {
            "total_bytes": encode(absent),
            "swap_total_bytes": encode(absent),
            "hugepages": encode(absent),
            "numa_distribution": encode(absent),
        }
    info = _meminfo()
    hugepage_size = info.get("Hugepagesize")
    return {
        "total_bytes": encode(info.get("MemTotal", unavailable("MemTotal absent"))),
        "swap_total_bytes": encode(info.get("SwapTotal", unavailable("SwapTotal absent"))),
        "hugepages": encode(
            f"{hugepage_size} bytes" if hugepage_size else unavailable("Hugepagesize absent")
        ),
        "numa_distribution": encode(_numa_memory()),
    }


# ----------------------------------------------------------------------- storage

_PSEUDO_FILESYSTEMS: Final[frozenset[str]] = frozenset(
    {
        "proc",
        "sysfs",
        "devtmpfs",
        "devpts",
        "tmpfs",
        "cgroup",
        "cgroup2",
        "securityfs",
        "pstore",
        "efivarfs",
        "bpf",
        "debugfs",
        "tracefs",
        "hugetlbfs",
        "mqueue",
        "fusectl",
        "configfs",
        "ramfs",
        "autofs",
        "binfmt_misc",
        "squashfs",
        "overlay",
        "nsfs",
    }
)


def _mounts() -> Iterator[tuple[str, str, str, str]]:
    raw = _read_text(_PROC / "mounts")
    if raw is None:
        return
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        device, mountpoint, fstype, options = parts[0], parts[1], parts[2], parts[3]
        if fstype in _PSEUDO_FILESYSTEMS or not device.startswith("/"):
            continue
        yield device, mountpoint, fstype, options


def _rotational(device: str) -> Measured[bool]:
    name = Path(device).name
    # Strip a partition suffix: nvme0n1p2 -> nvme0n1, sda3 -> sda.
    for candidate in (name, name.rstrip("0123456789").rstrip("p"), name.rstrip("0123456789")):
        flag = _read_int(_SYS / "block" / candidate / "queue/rotational")
        if flag is not None:
            return flag == 1
    return unavailable(f"no rotational flag for {device}")


def capture_storage() -> list[dict[str, Any]]:
    if not _is_linux():
        return []
    devices: list[dict[str, Any]] = []
    seen: set[str] = set()
    for device, mountpoint, fstype, options in _mounts():
        if device in seen:
            continue
        seen.add(device)
        entry: dict[str, Any] = {
            "device": device,
            "mountpoint": mountpoint,
            "filesystem": fstype,
            "mount_options": options,
            "rotational": encode(_rotational(device)),
        }
        try:
            stats = os.statvfs(mountpoint)
            entry["capacity_bytes"] = stats.f_blocks * stats.f_frsize
        except OSError:
            entry["capacity_bytes"] = encode(unavailable(f"statvfs failed for {mountpoint}"))
        devices.append(entry)
    return devices


# ---------------------------------------------------------------------- software


def _os_release_name() -> Measured[str]:
    raw = _read_text(Path("/etc/os-release"))
    if raw is None:
        return unavailable("/etc/os-release absent")
    for line in raw.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"')
    return unavailable("PRETTY_NAME absent from /etc/os-release")


def _tool_version(binary: str, *args: str) -> Measured[str]:
    if which(binary) is None:
        return unavailable(f"{binary} not on PATH")
    result = run_command([binary, *args])
    if result.timed_out:
        return unavailable(f"{binary} timed out")
    if not result.ok:
        return unavailable(f"{binary} exited {result.returncode}")
    line = first_line(result.stdout) or first_line(result.stderr)
    return line if line else unavailable(f"{binary} produced no version output")


def capture_software() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "os": encode(_os_release_name() if _is_linux() else platform.platform()),
        "kernel": encode(platform.release() or unavailable("platform.release() empty")),
        "libc": encode(
            f"{libc_name} {libc_version}".strip()
            if libc_name
            else unavailable("libc version not reported")
        ),
        "compiler": encode(_tool_version("cc", "--version")),
        "rust_toolchain": encode(_tool_version("rustc", "--version")),
        "python": encode(platform.python_version()),
        "postgres": encode(_tool_version("psql", "--version")),
        # There is no generic way to ask a host "is TheoDB installed"; the
        # adapter reports it once a system is bootstrapped.
        "theodb": encode(unavailable("reported by the system adapter, not by the host")),
        "container_runtime": encode(_tool_version("docker", "--version")),
        "benchmark_runner": f"theodb-bench {__version__}",
    }


# ------------------------------------------------------------------ capabilities


def capture_capabilities() -> dict[str, Any]:
    if not _is_linux():
        absent = _linux_only("host capabilities")
        return {
            key: encode(absent)
            for key in ("perf_events", "cgroup_v2", "cpu_affinity", "numa_control")
        }

    paranoid = _read_int(_PROC / "sys/kernel/perf_event_paranoid")
    perf_events: Measured[bool]
    if which("perf") is None:
        perf_events = unavailable("perf not on PATH")
    elif paranoid is None:
        perf_events = unavailable("perf_event_paranoid unreadable")
    else:
        # <= 2 permits per-process counters for an unprivileged user; 3 denies
        # them outright. Recording the reason matters: a run that silently lost
        # hardware counters would report them as absent without saying why.
        perf_events = paranoid <= 2

    return {
        "perf_events": encode(perf_events),
        "cgroup_v2": encode((_SYS / "fs/cgroup/cgroup.controllers").exists()),
        "cpu_affinity": encode(hasattr(os, "sched_setaffinity")),
        "numa_control": encode(which("numactl") is not None),
    }


# --------------------------------------------------------------- source control


def capture_source_control(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent
    if which("git") is None:
        absent = unavailable("git not on PATH")
        return {"benchmark_commit": encode(absent), "benchmark_dirty": encode(absent)}

    head = run_command(["git", "rev-parse", "HEAD"], cwd=root)
    status = run_command(["git", "status", "--porcelain"], cwd=root)
    commit: Measured[str] = (
        first_line(head.stdout) if head.ok else unavailable("git rev-parse failed")
    )
    dirty: Measured[bool] = (
        bool(status.stdout.strip()) if status.ok else unavailable("git status failed")
    )
    return {"benchmark_commit": encode(commit), "benchmark_dirty": encode(dirty)}


# ------------------------------------------------------------------------ facade


def capture_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """Capture the full environment record for a run bundle."""
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": platform.node() or "unknown",
        "cpu": capture_cpu(),
        "memory": capture_memory(),
        "storage": capture_storage(),
        "software": capture_software(),
        "capabilities": capture_capabilities(),
        "source_control": capture_source_control(repo_root),
    }
