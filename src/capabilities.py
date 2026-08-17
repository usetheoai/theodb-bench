"""Which capabilities any registered adapter can actually exercise.

The project declares fourteen capability surfaces. Crossing that vocabulary with
what the adapters declare is the only honest answer to "what can this harness
measure?", and doing it by hand is how the answer goes stale: the measurement
that opened B-073 found six of fourteen reachable, and a table typed that day
would still say six after the number moved.

So it is derived from the same registry the runs use. That has a second property
worth more than freshness: a capability nobody declares shows up as an empty row
rather than as an absent one. Absent reads as "not a thing"; empty reads as
"nothing reaches this yet", which is both true and the statement someone needs
before promising a pillar.

The `fake` adapter is excluded. It exists to test the harness, and counting it
would let the matrix report a pillar as reachable when no real system implements
it — the same shape of false green the rest of this project spends its effort
avoiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from theodb_bench.adapters.base import CAPABILITIES
from theodb_bench.registry import ADAPTERS

#: Adapters that exist to exercise the harness rather than a system.
_NOT_A_SYSTEM: Final[frozenset[str]] = frozenset({"fake"})


@dataclass(frozen=True)
class CapabilityRow:
    """One capability and the real adapters that declare it."""

    capability: str
    adapters: tuple[str, ...]

    @property
    def reachable(self) -> bool:
        return bool(self.adapters)


def capability_matrix() -> list[CapabilityRow]:
    """Every declared capability, in vocabulary order, with who reaches it."""
    declared: dict[str, list[str]] = {capability: [] for capability in CAPABILITIES}
    for name, entry in sorted(ADAPTERS.items()):
        if name in _NOT_A_SYSTEM:
            continue
        for capability in entry.factory().capabilities():
            if capability in declared:
                declared[capability].append(name)
    return [
        CapabilityRow(capability=capability, adapters=tuple(names))
        for capability, names in declared.items()
    ]


def render_capability_matrix() -> str:
    """The matrix as a Markdown table, for the README and for the terminal."""
    matrix = capability_matrix()
    reachable = sum(1 for row in matrix if row.reachable)

    lines = [
        f"**{reachable} of {len(CAPABILITIES)} capabilities are reachable by a real adapter.**",
        "",
        "| capability | adapters |",
        "|---|---|",
    ]
    for row in matrix:
        adapters = ", ".join(f"`{name}`" for name in row.adapters) if row.adapters else "—"
        lines.append(f"| `{row.capability}` | {adapters} |")
    lines += [
        "",
        "A dash is not a gap in this table — it is the measured state. `rerank`, "
        "`vectorizer` and `ai_sql` each reach an external model, and without an "
        "endpoint there is nothing to measure; a stub would put a number where an "
        "absence belongs.",
    ]
    return "\n".join(lines)
