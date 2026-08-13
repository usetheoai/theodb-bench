"""A deterministic fake system, and the failures a real one produces.

The runner is the thing under test here. Proving that it detects a crash, a
timeout, an escaped subprocess or a quality regression requires a system that
produces those on demand -- reliably, in milliseconds, with no database
installed (objective section 29).

Search is exact brute force over the loaded vectors, so recall is 1.0 by
construction unless a fault deliberately degrades it. That makes any recall
below 1.0 in a test a real finding about the runner, not noise from the fake.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from theodb_bench.adapters.base import (
    CAPABILITIES,
    BuildOutcome,
    Document,
    DocumentTableSpec,
    HybridQuery,
    IndexSpec,
    KnnQuery,
    KnnResult,
    LexicalQuery,
    LoadOutcome,
    RankedResult,
    SystemAdapter,
    VectorArray,
    VectorTableSpec,
    WriteOutcome,
)
from theodb_bench.errors import (
    AdapterError,
    ErrorContext,
    MeasurementError,
    Phase,
    SystemUnavailableError,
)


def _tokenise(text: str) -> list[str]:
    """Lower-case alphanumeric tokens. Deliberately simple and stated as such."""
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def _build_postings(documents: Sequence[Document]) -> dict[str, dict[int, int]]:
    postings: dict[str, dict[int, int]] = {}
    for document in documents:
        for term in _tokenise(document.text):
            postings.setdefault(term, {})
            postings[term][document.id] = postings[term].get(document.id, 0) + 1
    return postings


class Fault(str, Enum):
    """A failure mode the fake can be told to produce."""

    NONE = "none"
    SLOW = "slow"
    """Every query takes materially longer; the shape a regression has."""

    TIMEOUT = "timeout"
    """Queries exceed the declared budget and are counted as timeouts."""

    CRASH = "crash"
    """The system dies mid-run."""

    OOM = "oom"
    """The system is terminated the way an OOM kill terminates it."""

    INVALID_OUTPUT = "invalid_output"
    """Results come back structurally wrong -- wrong arity, impossible ids."""

    QUALITY_REGRESSION = "quality_regression"
    """Latency and throughput hold; the answers get worse. The failure a
    throughput-only benchmark cannot see."""

    ESCAPED_CHILD = "escaped_child"
    """A subprocess is spawned outside the declared resource controls."""

    NOT_READY = "not_ready"
    """The system never becomes ready."""


@dataclass
class FakeConfig:
    """How the fake should behave."""

    fault: Fault = Fault.NONE
    base_latency_seconds: float = 0.0002
    slow_multiplier: float = 25.0
    timeout_seconds: float = 0.5
    fail_after_queries: int = 5
    """How many queries succeed before a crash or OOM fault triggers."""

    quality_degradation: float = 0.5
    """Fraction of neighbours replaced by wrong ones under quality regression."""

    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "vector_exact": True,
            "vector_hnsw": True,
            "vector_ivfflat": False,
            "vector_filtered": True,
            "lexical": True,
            "hybrid": True,
            "rerank": True,
            "vectorizer": True,
        }
    )
    seed: int = 20260813
    rerank_latency_seconds: float = 0.001
    embed_seconds: float = 0.004
    """What the background worker spends per row. The freshness clock is a
    function of this and of how far behind the queue is."""

    worker_count: int = 1
    write_latency_seconds: float = 0.0002
    vectorizer_failure_every: int = 0
    """Fail every Nth job, to exercise retry accounting. 0 disables."""
    """A model that answers instantly would flatter any system that overlaps
    I/O with inference, so the mock has a declared non-zero cost."""


class FakeAdapter(SystemAdapter):
    """An in-process system that answers exactly and misbehaves on request."""

    system_id = "fake"

    def __init__(self, config: FakeConfig | None = None) -> None:
        self.config = config if config is not None else FakeConfig()
        self._vectors: dict[str, VectorArray] = {}
        self._metric: dict[str, str] = {}
        self._indexes: dict[str, IndexSpec] = {}
        self._search_parameters: dict[str, Any] = {}
        self._queries_served = 0
        self._running = False
        self._ready = False
        self._children: list[subprocess.Popen[bytes]] = []
        self._documents: dict[int, Document] = {}
        self._postings: dict[str, dict[int, int]] = {}
        self._queue: list[int] = []
        self._fresh: set[int] = set()
        self._worker_processed = 0
        self._worker_retries = 0
        self._worker_failures = 0
        self._worker_budget_seconds = 0.0
        self._worker_last_tick: float | None = None
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------ capabilities

    def capabilities(self) -> dict[str, bool]:
        return {name: self.config.capabilities.get(name, False) for name in CAPABILITIES}

    # --------------------------------------------------------------- lifecycle

    def prepare(self) -> None:
        self._queries_served = 0

    def start(self) -> None:
        self._running = True
        if self.config.fault is Fault.ESCAPED_CHILD:
            # A process the runner did not declare and does not control. This
            # is what the isolation check has to catch.
            self._children.append(
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        if self.config.fault is Fault.NOT_READY:
            raise SystemUnavailableError(
                "fake system never became ready",
                context=ErrorContext(phase=Phase.BOOTSTRAP, system=self.system_id),
            )
        if not self._running:
            raise SystemUnavailableError(
                "fake system was not started",
                context=ErrorContext(phase=Phase.BOOTSTRAP, system=self.system_id),
            )
        self._ready = True

    def stop(self) -> None:
        self._running = False
        self._ready = False
        for child in self._children:
            child.kill()
            child.wait(timeout=5)
        self._children.clear()

    def cleanup(self) -> None:
        self._vectors.clear()
        self._indexes.clear()

    # -------------------------------------------------------------------- data

    def load_dataset(self, spec: VectorTableSpec, vectors: VectorArray) -> LoadOutcome:
        self._require_ready(Phase.DATASET_LOAD)
        if vectors.ndim != 2 or vectors.shape[1] != spec.dimension:
            raise AdapterError(
                f"expected vectors of dimension {spec.dimension}, got shape {vectors.shape}",
                context=ErrorContext(phase=Phase.DATASET_LOAD, system=self.system_id),
            )
        started = time.perf_counter()
        # float32 on purpose: real vector columns are float4, and an oracle
        # that keeps float64 disagrees with the system on near-ties.
        self._vectors[spec.table] = np.ascontiguousarray(vectors, dtype=np.float32)
        self._metric[spec.table] = spec.metric
        return LoadOutcome(
            seconds=time.perf_counter() - started,
            rows_loaded=int(vectors.shape[0]),
            rows_expected=int(vectors.shape[0]),
        )

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        self._require_ready(Phase.INDEX_BUILD)
        self.require(index.capability, f"index kind {index.kind!r}")
        started = time.perf_counter()
        self._indexes[spec.table] = index
        vectors = self._vectors.get(spec.table)
        size = int(vectors.nbytes) if vectors is not None else 0
        return BuildOutcome(
            seconds=time.perf_counter() - started,
            index_size_bytes=size,
            parameters_in_force=dict(index.parameters),
        )

    def drop_indexes(self, spec: VectorTableSpec) -> None:
        self._indexes.pop(spec.table, None)

    def set_search_parameters(self, parameters: dict[str, Any]) -> None:
        self._search_parameters = dict(parameters)

    # ----------------------------------------------------------------- queries

    def execute(self, query: KnnQuery) -> KnnResult:
        self._require_ready(Phase.MEASUREMENT)
        self._maybe_fail()
        vectors = self._vectors.get(query.table)
        if vectors is None:
            raise AdapterError(
                f"table {query.table!r} was never loaded",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )

        started = time.perf_counter()
        ids, distances = self._exact_search(vectors, query)
        if self.config.fault is Fault.QUALITY_REGRESSION:
            ids, distances = self._degrade(ids, distances, vectors.shape[0])
        if self.config.fault is Fault.INVALID_OUTPUT:
            ids = ids[:-1] if len(ids) > 1 else (2**62,)
        self._sleep_for_fault()
        elapsed = time.perf_counter() - started

        if self.config.fault is Fault.TIMEOUT:
            raise MeasurementError(
                f"query exceeded the {self.config.timeout_seconds}s budget",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        self._queries_served += 1
        return KnnResult(ids=tuple(ids), distances=tuple(distances), latency_seconds=elapsed)

    def _exact_search(
        self, vectors: VectorArray, query: KnnQuery
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        probe = np.ascontiguousarray(query.vector, dtype=np.float32)
        metric = self._metric.get(query.table, query.metric)
        if metric == "l2":
            deltas = vectors.astype(np.float64) - probe.astype(np.float64)
            scores = np.einsum("ij,ij->i", deltas, deltas)
        elif metric == "ip":
            scores = -(vectors.astype(np.float64) @ probe.astype(np.float64))
        elif metric == "cosine":
            left = vectors.astype(np.float64)
            norms = np.linalg.norm(left, axis=1) * float(np.linalg.norm(probe.astype(np.float64)))
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = 1.0 - (left @ probe.astype(np.float64)) / norms
            scores = np.nan_to_num(scores, nan=2.0)
        else:
            raise AdapterError(
                f"unknown metric {metric!r}",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        k = min(query.k, scores.shape[0])
        # Deterministic tie-breaking by id: without it, equal distances resolve
        # in whatever order the sort happens to produce.
        order = np.lexsort((np.arange(scores.shape[0]), scores))[:k]
        return tuple(int(i) for i in order), tuple(float(scores[i]) for i in order)

    def _degrade(
        self, ids: tuple[int, ...], distances: tuple[float, ...], corpus_size: int
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Replace some correct neighbours with wrong ones, keeping latency."""
        count = max(1, int(len(ids) * self.config.quality_degradation))
        wrong = self._rng.integers(0, corpus_size, size=count)
        mutated = list(ids)
        for position, replacement in enumerate(wrong):
            mutated[len(ids) - 1 - position] = int(replacement)
        return tuple(mutated), distances

    # ---------------------------------------------------------------- retrieval

    def load_documents(self, spec: DocumentTableSpec, documents: Sequence[Document]) -> LoadOutcome:
        self._require_ready(Phase.DATASET_LOAD)
        started = time.perf_counter()
        self._documents = {document.id: document for document in documents}
        self._postings = _build_postings(documents)
        vectors = np.vstack([document.vector for document in documents]).astype(np.float32)
        self._vectors[spec.table] = vectors
        self._metric[spec.table] = spec.metric
        return LoadOutcome(
            seconds=time.perf_counter() - started,
            rows_loaded=len(self._documents),
            rows_expected=len(documents),
        )

    def execute_lexical(self, query: LexicalQuery) -> RankedResult:
        """Deterministic term-frequency scoring.

        Not BM25: this exists to exercise the pipeline, and calling it BM25
        would invite someone to compare its numbers with a real one.
        """
        self.require("lexical")
        self._require_ready(Phase.MEASUREMENT)
        started = time.perf_counter()
        scores: dict[int, float] = {}
        for term in _tokenise(query.text):
            for doc_id, count in self._postings.get(term, {}).items():
                scores[doc_id] = scores.get(doc_id, 0.0) + float(count)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: query.n]
        self._sleep_for_fault()
        return RankedResult(
            ids=tuple(doc_id for doc_id, _ in ranked),
            scores=tuple(score for _, score in ranked),
            latency_seconds=time.perf_counter() - started,
            stage_seconds={"lexical": time.perf_counter() - started},
        )

    def execute_hybrid(self, query: HybridQuery) -> RankedResult:
        """Fuse the two legs the way the system would, with stage timings."""
        self.require("hybrid")
        self._require_ready(Phase.MEASUREMENT)
        from theodb_bench.analysis.fusion import fuse_to_ids

        lexical_started = time.perf_counter()
        lexical = self.execute_lexical(LexicalQuery(query.table, query.text, query.n))
        lexical_seconds = time.perf_counter() - lexical_started

        vector_started = time.perf_counter()
        dense = self.execute(
            KnnQuery(table=query.table, vector=query.vector, k=query.n, metric=query.metric)
        )
        vector_seconds = time.perf_counter() - vector_started

        fusion_started = time.perf_counter()
        fused = fuse_to_ids(
            {"lexical": list(lexical.ids), "vector": list(dense.ids)},
            n=query.n,
            k=query.rrf_k,
            weights=query.weights or None,
        )
        fusion_seconds = time.perf_counter() - fusion_started

        return RankedResult(
            ids=tuple(fused),
            scores=tuple(float(len(fused) - position) for position in range(len(fused))),
            latency_seconds=lexical_seconds + vector_seconds + fusion_seconds,
            stage_seconds={
                "lexical": lexical_seconds,
                "vector": vector_seconds,
                "fusion": fusion_seconds,
            },
        )

    def execute_rerank(self, text: str, candidate_ids: Sequence[int]) -> RankedResult:
        """Reorder candidates, charging the model's time to a separate stage.

        The model here is a fixed function of the text and the document, which
        keeps the pipeline deterministic. Its declared latency is non-zero
        because a zero-latency model changes the concurrency regime of the
        whole loop.
        """
        self.require("rerank")
        self._require_ready(Phase.MEASUREMENT)
        database_started = time.perf_counter()
        terms = set(_tokenise(text))
        database_seconds = time.perf_counter() - database_started

        model_started = time.perf_counter()
        scored: list[tuple[int, float]] = []
        for doc_id in candidate_ids:
            document = self._documents.get(doc_id)
            overlap = len(terms & set(_tokenise(document.text))) if document else 0
            scored.append((doc_id, float(overlap)))
        time.sleep(self.config.rerank_latency_seconds)
        model_seconds = time.perf_counter() - model_started

        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        return RankedResult(
            ids=tuple(doc_id for doc_id, _ in ranked),
            scores=tuple(score for _, score in ranked),
            latency_seconds=database_seconds + model_seconds,
            stage_seconds={"database": database_seconds, "model": model_seconds},
        )

    # --------------------------------------------------------------- operations

    def insert_document(self, spec: DocumentTableSpec, document: Document) -> WriteOutcome:
        """A foreground write. The embedding is queued, not computed here.

        That is the whole design under test: the transaction returns before the
        derived embedding exists, so the write looks fast and the freshness
        clock has to be measured separately.
        """
        self.require("vectorizer")
        self._require_ready(Phase.MEASUREMENT)
        started = time.perf_counter()
        self._documents[document.id] = document
        for term in _tokenise(document.text):
            self._postings.setdefault(term, {})
            self._postings[term][document.id] = self._postings[term].get(document.id, 0) + 1
        self._queue.append(document.id)
        self._fresh.discard(document.id)
        time.sleep(self.config.write_latency_seconds)
        return WriteOutcome(row_id=document.id, latency_seconds=time.perf_counter() - started)

    def update_document_text(self, spec: DocumentTableSpec, row_id: int, text: str) -> WriteOutcome:
        self.require("vectorizer")
        self._require_ready(Phase.MEASUREMENT)
        existing = self._documents.get(row_id)
        if existing is None:
            raise AdapterError(
                f"row {row_id} does not exist",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        started = time.perf_counter()
        self._documents[row_id] = Document(id=row_id, text=text, vector=existing.vector)
        # Changing the source invalidates the derived embedding, which is what
        # makes an update a harder case than an insert.
        self._fresh.discard(row_id)
        self._queue.append(row_id)
        time.sleep(self.config.write_latency_seconds)
        return WriteOutcome(row_id=row_id, latency_seconds=time.perf_counter() - started)

    def is_fresh(self, spec: DocumentTableSpec, row_id: int) -> bool:
        self.require("vectorizer")
        self._drain_worker()
        return row_id in self._fresh

    def queue_depth(self) -> int:
        self.require("vectorizer")
        self._drain_worker()
        return len(self._queue)

    def vectorizer_stats(self) -> dict[str, Any]:
        self.require("vectorizer")
        self._drain_worker()
        return {
            "queue_depth": len(self._queue),
            "processed": self._worker_processed,
            "retries": self._worker_retries,
            "failures": self._worker_failures,
            "workers": self.config.worker_count,
        }

    def _drain_worker(self) -> None:
        """Advance the background worker by the wall time that has passed.

        Modelled rather than threaded: a real worker consumes the queue at a
        rate, and a simulated one that consumed it instantly would make every
        freshness measurement zero.
        """
        now = time.perf_counter()
        if self._worker_last_tick is None:
            self._worker_last_tick = now
            return
        elapsed = now - self._worker_last_tick
        self._worker_last_tick = now
        self._worker_budget_seconds += elapsed * self.config.worker_count

        per_row = self.config.embed_seconds
        while self._queue and self._worker_budget_seconds >= per_row:
            self._worker_budget_seconds -= per_row
            row_id = self._queue.pop(0)
            self._worker_processed += 1
            failure_every = self.config.vectorizer_failure_every
            if failure_every and self._worker_processed % failure_every == 0:
                # A failed job goes back on the queue and is counted, rather
                # than silently disappearing and leaving the row stale forever.
                self._worker_retries += 1
                self._queue.append(row_id)
                continue
            self._fresh.add(row_id)

    # ------------------------------------------------------------------ faults

    def _sleep_for_fault(self) -> None:
        latency = self.config.base_latency_seconds
        if self.config.fault is Fault.SLOW:
            latency *= self.config.slow_multiplier
        elif self.config.fault is Fault.TIMEOUT:
            latency = self.config.timeout_seconds * 1.1
        if latency > 0:
            time.sleep(latency)

    def _maybe_fail(self) -> None:
        if self._queries_served < self.config.fail_after_queries:
            return
        if self.config.fault is Fault.CRASH:
            self._running = False
            self._ready = False
            raise SystemUnavailableError(
                "fake system crashed",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )
        if self.config.fault is Fault.OOM:
            self._running = False
            self._ready = False
            raise SystemUnavailableError(
                "fake system terminated: out of memory",
                context=ErrorContext(
                    phase=Phase.MEASUREMENT,
                    system=self.system_id,
                    details={"oom": True},
                ),
            )

    def _require_ready(self, phase: Phase) -> None:
        if not self._ready:
            raise SystemUnavailableError(
                "fake system is not ready",
                context=ErrorContext(phase=phase, system=self.system_id),
            )

    # ------------------------------------------------------------------ config

    def collect_stats(self) -> dict[str, Any]:
        return {
            "queries_served": self._queries_served,
            "tables": sorted(self._vectors),
            "indexes": {table: index.label() for table, index in self._indexes.items()},
        }

    def export_config(self) -> dict[str, Any]:
        return {
            "version": "fake-1",
            "effective_configuration": {
                "fault": self.config.fault.value,
                "base_latency_seconds": self.config.base_latency_seconds,
                "seed": self.config.seed,
            },
            "index_configuration": {table: index.label() for table, index in self._indexes.items()},
        }
