# Hardware

What a host must be for its numbers to mean anything.

## Two classes of machine

**A development machine** runs `smoke` and `research`. It produces real
measurements that are never publishable evidence. This is not a formality: a
laptop under frequency scaling, with swap enabled and a browser open, measures
its own afternoon.

**A benchmark host** runs `pr`, `nightly` and `release`. It is dedicated,
known, and controlled.

## Requirements by profile

`doctor` enforces these; the table is what it enforces.

| Check | pr / nightly | release |
|---|---|---|
| OS and kernel identified | mandatory | mandatory |
| CPU topology readable | mandatory | mandatory |
| At least 4 logical CPUs | mandatory | mandatory |
| Memory total readable | mandatory | mandatory |
| Real block device present | mandatory | mandatory |
| cgroup v2 mounted | mandatory | mandatory |
| CPU affinity settable | mandatory | mandatory |
| `git` present | mandatory | mandatory |
| CPU governor is `performance` | advisory | **mandatory** |
| Swap disabled | advisory | **mandatory** |
| NUMA placement controllable | advisory | **mandatory** (multi-node hosts) |
| Hardware counters available | advisory | advisory |

A warning that is mandatory for a profile blocks that profile. A release
measured under frequency scaling is a methodology defect, not a note to read
later.

## Preparing a benchmark host

```bash
# Frequency scaling off.
sudo cpupower frequency-set --governor performance

# Swap off: paging silently distorts tail latency.
sudo swapoff -a

# Confirm before measuring anything.
theodb-bench doctor --profile release
```

Exit code 2 means the host may not run that profile. The blocking checks are
listed and marked with `*`.

## Exclusivity

Two benchmarks sharing a host measure each other. The runner takes a
whole-host advisory lock, and the dedicated CI workflow pins concurrency to one
run at a time.

Background services matter too. A backup job, an indexer or an update daemon
that wakes during a measurement window is a variable nobody recorded.

## Hardware class

Regression comparison keys on a `hardware_class` string. Two runs from
different classes are `INCOMPARABLE` — never silently compared. Give a class a
name that changes when the machine does.

## What the environment capture records

CPU vendor, model, sockets, physical cores, logical CPUs, SMT state, frequency
governor, NUMA nodes and cache hierarchy; memory total, swap and per-node
distribution; block devices with filesystem and mount options; OS, kernel,
libc, compiler, toolchains and container runtime; and whether perf events,
cgroup v2, CPU affinity and NUMA control are actually available.

Microarchitecture is deliberately **not** derived from family/model numbers. A
wrong value there makes two runs look comparable when they are not.
