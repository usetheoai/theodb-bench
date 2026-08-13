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


@dataclass(frozen=True)
class Document:
    """A corpus document: text for the lexical leg, a vector for the dense leg."""

    id: int
    text: str
    vector: VectorArray


@dataclass(frozen=True)
class DocumentTableSpec:
    """A table holding documents for retrieval."""

    table: str
    dimension: int
    metric: Metric = "cosine"
    text_column: str = "content"
    embedding_column: str = "embedding"


@dataclass(frozen=True)
class LexicalQuery:
    """A keyword request against the lexical leg."""

    table: str
    text: str
    n: int


@dataclass(frozen=True)
class HybridQuery:
    """A request the system fuses internally.

    The benchmark also fuses the individual legs itself, so the system's own
    fusion can be checked rather than trusted.
    """

    table: str
    text: str
    vector: VectorArray
    n: int
    metric: Metric = "cosine"
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedResult:
    """An ordered list of document ids, and how long it took to produce."""

    ids: tuple[int, ...]
    scores: tuple[float, ...]
    latency_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
    """Per-stage breakdown. Model inference, when involved, is a separate stage
    so it is never folded into the database's time."""


@dataclass(frozen=True)
class WriteOutcome:
    """One foreground write, timed as the writing transaction experienced it.

    This is the first of the two clocks an operations benchmark measures. The
    second -- how long until the derived embedding is queryable -- is observed
    separately, because a system that moves embedding work out of the
    transaction improves this number while leaving the other one to be checked.
    """

    row_id: int
    latency_seconds: float
    accepted: bool = True


@dataclass(frozen=True)
class GraphSpec:
    """A graph to load and traverse."""

    name: str
    directed: bool = True
    edge_type: str | None = None


@dataclass(frozen=True)
class TraversalQuery:
    """A k-hop neighbourhood expansion from one source vertex."""

    graph: str
    source: int
    hops: int
    limit: int | None = None
    edge_type: str | None = None


@dataclass(frozen=True)
class TraversalResult:
    """Vertices reached, and what it cost to reach them.

    ``edges_visited`` is the work done, not the answer size. A traversal that
    returns few vertices after walking many edges is expensive, and reporting
    only the result size would hide that.
    """

    vertices: tuple[int, ...]
    edges_visited: int
    latency_seconds: float


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

    # ---------------------------------------------------------- retrieval

    def load_documents(self, spec: DocumentTableSpec, documents: Sequence[Document]) -> LoadOutcome:
        """Load a document corpus. Systems without a lexical leg do not have one."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} cannot load a document corpus",
            context=ErrorContext(phase=Phase.DATASET_LOAD, system=self.system_id),
        )

    def execute_lexical(self, query: LexicalQuery) -> RankedResult:
        """Keyword retrieval."""
        self.require("lexical")
        raise UnsupportedCapabilityError(
            f"{self.system_id} declares lexical support but does not implement it",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def execute_hybrid(self, query: HybridQuery) -> RankedResult:
        """Retrieval fused by the system itself."""
        self.require("hybrid")
        raise UnsupportedCapabilityError(
            f"{self.system_id} declares hybrid support but does not implement it",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def execute_rerank(self, text: str, candidate_ids: Sequence[int]) -> RankedResult:
        """Reorder candidates with a cross-encoder or equivalent.

        Model latency belongs in `stage_seconds`, separate from database time.
        """
        self.require("rerank")
        raise UnsupportedCapabilityError(
            f"{self.system_id} declares rerank support but does not implement it",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    # ---------------------------------------------------------- operations

    def insert_document(self, spec: DocumentTableSpec, document: Document) -> WriteOutcome:
        """Write one row in the foreground, returning the transaction's own latency."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} does not support foreground document writes",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def update_document_text(self, spec: DocumentTableSpec, row_id: int, text: str) -> WriteOutcome:
        """Change a source column, which should invalidate any derived embedding."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} does not support source updates",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def is_fresh(self, spec: DocumentTableSpec, row_id: int) -> bool:
        """Whether the row's derived embedding reflects its current source.

        The second clock stops when this turns true.
        """
        raise UnsupportedCapabilityError(
            f"{self.system_id} does not expose embedding freshness",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def queue_depth(self) -> int:
        """Rows waiting for the background vectorizer."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} does not expose a vectorizer queue",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def vectorizer_stats(self) -> dict[str, Any]:
        """Worker counters: processed, retries, failures."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} does not expose vectorizer statistics",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    # ---------------------------------------------------------------- graph

    def load_graph(
        self, spec: GraphSpec, edges: Sequence[tuple[int, int]], vertex_count: int
    ) -> BuildOutcome:
        """Build the graph structure. Timed as the build, not as a query."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} cannot load a graph",
            context=ErrorContext(phase=Phase.INDEX_BUILD, system=self.system_id),
        )

    def traverse(self, query: TraversalQuery) -> TraversalResult:
        """Expand a neighbourhood."""
        self.require("graph")
        raise UnsupportedCapabilityError(
            f"{self.system_id} declares graph support but does not implement traversal",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

    def graph_stats(self) -> dict[str, Any]:
        """Structure size, for bytes-per-edge accounting."""
        raise UnsupportedCapabilityError(
            f"{self.system_id} does not expose graph statistics",
            context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
        )

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
