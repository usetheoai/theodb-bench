"""The system adapter contract.

A benchmark suite must not contain system-specific lifecycle logic (TRD D2).
Everything a particular database needs in order to be measured -- how it starts,
how data is loaded, how an index is built, how a query is issued -- lives behind
this interface, so adding a system does not touch the core.

Two rules the contract exists to enforce:

An adapter declares its capabilities, and a workload feature the adapter does
not support produces an explicit ``unsupported`` result rather than a failure or
a fabricated number (TRD §27).

An adapter exports the configuration that was actually in force, read back from
the system rather than echoed from what was requested. A comparison is only
fair if the reader can see what each side was actually running with (TRD §28).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt
from theodb_bench.errors import (
    AdapterError,
    ErrorContext,
    Phase,
    UnsupportedCapabilityError,
)

SYSTEM_SCHEMA_VERSION: Final[int] = 1

# Capability names are a closed vocabulary so that two adapters cannot disagree
# about what "hybrid" means by spelling it differently.
CAPABILITIES: Final[tuple[str, ...]] = (
    "vector_exact",
    "vector_hnsw",
    "vector_ivfflat",
    "vector_quantized",
    "vector_filtered",
    "lexical",
    "hybrid",
    "rerank",
    "columnar",
    "parquet",
    "graph",
    "vectorizer",
    "ai_sql",
)

Metric = str
METRICS: Final[tuple[Metric, ...]] = ("l2", "cosine", "ip")

VectorArray: TypeAlias = npt.NDArray[np.floating[Any]]
"""A 2-D corpus or a 1-D probe; float32 or float64 at the boundary."""


@dataclass(frozen=True)
class VectorTableSpec:
    """A table of vectors to load."""

    table: str
    dimension: int
    metric: Metric = "l2"
    embedding_column: str = "embedding"


@dataclass(frozen=True)
class IndexSpec:
    """An index to build. ``kind`` of ``none`` means exact search."""

    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def capability(self) -> str:
        return {
            "none": "vector_exact",
            "hnsw": "vector_hnsw",
            "ivfflat": "vector_ivfflat",
        }.get(self.kind, f"vector_{self.kind}")

    def label(self) -> str:
        if not self.parameters:
            return self.kind
        rendered = " ".join(f"{key}={value}" for key, value in sorted(self.parameters.items()))
        return f"{self.kind} {rendered}"


@dataclass(frozen=True)
class KnnQuery:
    """One nearest-neighbour request."""

    table: str
    vector: VectorArray
    k: int
    metric: Metric = "l2"
    search_parameters: dict[str, Any] = field(default_factory=dict)
    tenant: str | None = None
    """Filter value, when the workload exercises filtered retrieval."""


@dataclass(frozen=True)
class KnnResult:
    """What the system returned, and how long it took.

    ``latency_seconds`` is measured by the adapter around the system call
    itself, so that client-side overhead is not attributed to the database.
    """

    ids: tuple[int, ...]
    distances: tuple[float, ...]
    latency_seconds: float


@dataclass(frozen=True)
class BuildOutcome:
    """The cost of building an index."""

    seconds: float
    index_size_bytes: int | None = None
    parameters_in_force: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadOutcome:
    """The cost of loading data, and proof that all of it arrived."""

    seconds: float
    rows_loaded: int
    rows_expected: int

    @property
    def complete(self) -> bool:
        return self.rows_loaded == self.rows_expected


class SystemAdapter(ABC):
    """Everything the runner needs in order to measure one system."""

    system_id: str

    # ------------------------------------------------------------ capabilities

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Declared support per workload feature."""

    def supports(self, capability: str) -> bool:
        if capability not in CAPABILITIES:
            raise AdapterError(
                f"unknown capability {capability!r}; known: {', '.join(CAPABILITIES)}",
                context=ErrorContext(phase=Phase.BOOTSTRAP, system=self.system_id),
            )
        return self.capabilities().get(capability, False)

    def require(self, capability: str, detail: str = "") -> None:
        """Refuse a workload feature this system does not have.

        Raises ``UnsupportedCapabilityError``, which the orchestrator records as
        an unsupported result rather than as a failed run.
        """
        if not self.supports(capability):
            suffix = f": {detail}" if detail else ""
            raise UnsupportedCapabilityError(
                f"{self.system_id} does not support {capability}{suffix}",
                context=ErrorContext(
                    phase=Phase.BOOTSTRAP,
                    system=self.system_id,
                    details={"capability": capability},
                ),
            )

    # --------------------------------------------------------------- lifecycle

    @abstractmethod
    def prepare(self) -> None:
        """Everything that must happen before the system starts."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        """Block until the system accepts work, or raise SystemUnavailableError."""

    @abstractmethod
    def load_dataset(self, spec: VectorTableSpec, vectors: VectorArray) -> LoadOutcome: ...

    @abstractmethod
    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome: ...

    @abstractmethod
    def execute(self, query: KnnQuery) -> KnnResult: ...

    @abstractmethod
    def collect_stats(self) -> dict[str, Any]:
        """System-specific statistics for the run bundle."""

    @abstractmethod
    def export_config(self) -> dict[str, Any]:
        """The configuration actually in force, read back from the system."""

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    # ------------------------------------------------------------- convenience

    def set_search_parameters(self, parameters: dict[str, Any]) -> None:
        """Apply per-query tuning.

        Optional by design, not by oversight: a system with no per-query
        knobs has nothing to set, and forcing it to implement a stub would
        add a method that lies about being meaningful.
        """
        return None

    def drop_indexes(self, spec: VectorTableSpec) -> None:
        """Remove indexes left by other configurations.

        Measuring one index while another exists on the same column lets the
        planner choose between them, and one sweep silently flattens onto the
        other. Adapters with no persistent index have nothing to drop.
        """
        return None

    def system_payload(self) -> dict[str, Any]:
        """The ``system.json`` artifact for this run."""
        payload: dict[str, Any] = {
            "schema_version": SYSTEM_SCHEMA_VERSION,
            "system": self.system_id,
            "capabilities": {name: self.supports(name) for name in CAPABILITIES},
        }
        payload.update(self.export_config())
        return payload

    # ------------------------------------------------------------ context mgmt

    def __enter__(self) -> SystemAdapter:
        self.prepare()
        self.start()
        self.wait_ready()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self.stop()
        finally:
            self.cleanup()


def execute_batch(adapter: SystemAdapter, queries: Sequence[KnnQuery]) -> list[KnnResult]:
    """Run a batch of queries in order, one at a time.

    Deliberately unclever: no batching, no pipelining, no reordering. Anything
    smarter would change what is being measured.
    """
    return [adapter.execute(query) for query in queries]
