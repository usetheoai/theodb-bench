"""System adapters.

Adding a system means adding a module here. Nothing in the core imports a
concrete adapter by name; the registry resolves them (TRD D2).
"""

from __future__ import annotations

from theodb_bench.adapters.base import (
    BuildOutcome,
    IndexSpec,
    KnnQuery,
    KnnResult,
    LoadOutcome,
    SystemAdapter,
    VectorTableSpec,
)

__all__ = [
    "BuildOutcome",
    "IndexSpec",
    "KnnQuery",
    "KnnResult",
    "LoadOutcome",
    "SystemAdapter",
    "VectorTableSpec",
]
