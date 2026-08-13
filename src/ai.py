"""Model endpoints for AI SQL and reranking workloads.

External inference is the single largest reproducibility risk in this project.
A hosted model changes underneath a benchmark without telling anyone, so a
number measured through one describes a moment rather than a system.

Three modes, with different standings:

``mock`` is deterministic and is the only mode a regression gate may use. Its
latency profile is **declared and non-zero**: a model that answers instantly
changes the concurrency regime of the whole loop and flatters any system that
overlaps I/O with inference.

``local`` runs a frozen model the operator controls. Reproducible if and only
if the weights and parameters are pinned, which the run records.

``remote`` calls a hosted endpoint. Always marked environment-dependent, and
**never** permitted to back a hard regression gate.

Whatever the mode, the report separates database time, network time and
inference time. A composite that silently includes inference measures the model
vendor.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.errors import ConfigError, ErrorContext, Phase

DEFAULT_MOCK_LATENCY_SECONDS: Final[float] = 0.005
"""Non-zero on purpose. See the module docstring."""


class EndpointMode(str, Enum):
    MOCK = "mock"
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class ModelCall:
    """One inference, with its cost attributed."""

    output: str
    inference_seconds: float
    network_seconds: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_seconds(self) -> float:
        return self.inference_seconds + self.network_seconds


@dataclass(frozen=True)
class EndpointDescriptor:
    """What a run records about the model it used.

    Without this a result cannot be reproduced or even understood later: "we
    used an LLM" is not a measurement condition.
    """

    mode: EndpointMode
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    network_placement: str = "in-process"
    environment_dependent: bool = False
    gate_eligible: bool = False
    """Whether a hard regression gate may be built on this endpoint."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "model": self.model,
            "parameters": dict(self.parameters),
            "batch_size": self.batch_size,
            "network_placement": self.network_placement,
            "environment_dependent": self.environment_dependent,
            "gate_eligible": self.gate_eligible,
        }


class ModelEndpoint(ABC):
    """A model the benchmark can call."""

    @abstractmethod
    def descriptor(self) -> EndpointDescriptor: ...

    @abstractmethod
    def generate(self, prompt: str) -> ModelCall: ...

    @abstractmethod
    def embed(self, text: str, dimension: int) -> tuple[npt.NDArray[np.float32], ModelCall]: ...

    def generate_batch(self, prompts: Sequence[str]) -> list[ModelCall]:
        """Batched generation.

        The default is a loop, which is honest: an endpoint that does not
        actually batch must not appear to, or the benchmark would credit it
        with an efficiency it does not have.
        """
        return [self.generate(prompt) for prompt in prompts]


class MockEndpoint(ModelEndpoint):
    """A deterministic endpoint for measuring database and control-plane cost.

    The output is a pure function of the prompt, so two runs produce identical
    text and any difference in a downstream quality metric comes from the
    database rather than from the model.
    """

    def __init__(
        self,
        *,
        latency_seconds: float = DEFAULT_MOCK_LATENCY_SECONDS,
        batch_size: int = 1,
        batch_efficiency: float = 0.5,
    ) -> None:
        if latency_seconds <= 0:
            # A zero-latency model is not a faster model; it is a different
            # concurrency regime, and measuring under it flatters any system
            # that overlaps I/O with inference.
            raise ConfigError(
                "the mock endpoint's latency must be non-zero; a model that answers "
                "instantly changes the concurrency regime of the whole loop",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if not 0.0 < batch_efficiency <= 1.0:
            raise ConfigError(
                f"batch efficiency must be in (0, 1], got {batch_efficiency}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        self.latency_seconds = latency_seconds
        self.batch_size = batch_size
        self.batch_efficiency = batch_efficiency

    def descriptor(self) -> EndpointDescriptor:
        return EndpointDescriptor(
            mode=EndpointMode.MOCK,
            model="mock-deterministic-1",
            parameters={
                "latency_seconds": self.latency_seconds,
                "batch_efficiency": self.batch_efficiency,
            },
            batch_size=self.batch_size,
            network_placement="in-process",
            environment_dependent=False,
            gate_eligible=True,
        )

    def generate(self, prompt: str) -> ModelCall:
        started = time.perf_counter()
        time.sleep(self.latency_seconds)
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return ModelCall(
            output=f"mock:{digest[:16]}",
            inference_seconds=elapsed,
            network_seconds=0.0,
            prompt_tokens=len(prompt.split()),
            completion_tokens=4,
        )

    def generate_batch(self, prompts: Sequence[str]) -> list[ModelCall]:
        """Batched generation with a declared efficiency factor.

        Real batching amortises fixed cost; the factor makes that visible and
        tunable rather than implied.
        """
        if not prompts:
            return []
        started = time.perf_counter()
        time.sleep(self.latency_seconds * (1 + (len(prompts) - 1) * self.batch_efficiency))
        elapsed = time.perf_counter() - started
        per_call = elapsed / len(prompts)
        return [
            ModelCall(
                output=f"mock:{hashlib.sha256(p.encode('utf-8')).hexdigest()[:16]}",
                inference_seconds=per_call,
                network_seconds=0.0,
                prompt_tokens=len(p.split()),
                completion_tokens=4,
            )
            for p in prompts
        ]

    def embed(self, text: str, dimension: int) -> tuple[npt.NDArray[np.float32], ModelCall]:
        """A deterministic embedding derived from the text.

        Seeded by the text digest, so the same text always embeds identically
        and similar texts do **not** embed similarly. That is a deliberate
        limitation: the mock measures cost, never semantic quality.
        """
        call = self.generate(text)
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(dimension).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector, call


class LocalEndpoint(ModelEndpoint):
    """A frozen model the operator runs and pins.

    Reproducible only to the extent that the weights and parameters are pinned,
    which is why the descriptor carries both. This class is a seam: the actual
    call is delegated, so no inference library is a dependency of the runner.
    """

    def __init__(
        self,
        model: str,
        call: Any,
        *,
        parameters: dict[str, Any] | None = None,
        batch_size: int = 1,
    ) -> None:
        if not model:
            raise ConfigError(
                "a local endpoint must name the model it pins",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        self.model = model
        self._call = call
        self.parameters = parameters or {}
        self.batch_size = batch_size

    def descriptor(self) -> EndpointDescriptor:
        return EndpointDescriptor(
            mode=EndpointMode.LOCAL,
            model=self.model,
            parameters=dict(self.parameters),
            batch_size=self.batch_size,
            network_placement="localhost",
            environment_dependent=False,
            gate_eligible=False,
        )

    def generate(self, prompt: str) -> ModelCall:
        started = time.perf_counter()
        output = self._call(prompt)
        elapsed = time.perf_counter() - started
        return ModelCall(
            output=str(output),
            inference_seconds=elapsed,
            network_seconds=0.0,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(str(output).split()),
        )

    def embed(self, text: str, dimension: int) -> tuple[npt.NDArray[np.float32], ModelCall]:
        raise ConfigError(
            "this local endpoint was configured for generation, not embedding",
            context=ErrorContext(phase=Phase.MEASUREMENT),
        )


class RemoteEndpoint(ModelEndpoint):
    """A hosted endpoint. Environment-dependent, never gate-eligible.

    No credential ever reaches this class. The caller supplies an already
    authenticated callable, so there is no path by which a key could be logged,
    echoed or written into a result bundle.
    """

    def __init__(
        self,
        model: str,
        call: Any,
        *,
        parameters: dict[str, Any] | None = None,
        network_placement: str = "unknown",
        batch_size: int = 1,
    ) -> None:
        if not model:
            raise ConfigError(
                "a remote endpoint must record which model was called",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        self.model = model
        self._call = call
        self.parameters = parameters or {}
        self.network_placement = network_placement
        self.batch_size = batch_size

    def descriptor(self) -> EndpointDescriptor:
        return EndpointDescriptor(
            mode=EndpointMode.REMOTE,
            model=self.model,
            parameters=dict(self.parameters),
            batch_size=self.batch_size,
            network_placement=self.network_placement,
            environment_dependent=True,
            gate_eligible=False,
        )

    def generate(self, prompt: str) -> ModelCall:
        started = time.perf_counter()
        result = self._call(prompt)
        elapsed = time.perf_counter() - started
        # A remote call's inference and network time cannot be separated from
        # the client side, so the whole cost is attributed to the network and
        # the report says the split is unavailable rather than inventing one.
        return ModelCall(
            output=str(result),
            inference_seconds=0.0,
            network_seconds=elapsed,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(str(result).split()),
        )

    def embed(self, text: str, dimension: int) -> tuple[npt.NDArray[np.float32], ModelCall]:
        raise ConfigError(
            "this remote endpoint was configured for generation, not embedding",
            context=ErrorContext(phase=Phase.MEASUREMENT),
        )


@dataclass
class CostBreakdown:
    """Where a pipeline's time went. The whole point of the AI SQL surface."""

    database_seconds: float = 0.0
    network_seconds: float = 0.0
    inference_seconds: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, call: ModelCall) -> None:
        self.network_seconds += call.network_seconds
        self.inference_seconds += call.inference_seconds
        self.calls += 1
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens

    @property
    def total_seconds(self) -> float:
        return self.database_seconds + self.network_seconds + self.inference_seconds

    @property
    def database_share(self) -> float | None:
        """Fraction of the time the database was responsible for."""
        total = self.total_seconds
        return self.database_seconds / total if total > 0 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "database_seconds": self.database_seconds,
            "network_seconds": self.network_seconds,
            "inference_seconds": self.inference_seconds,
            "total_seconds": self.total_seconds,
            "database_share": self.database_share,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def require_gate_eligible(endpoint: ModelEndpoint) -> None:
    """Refuse to build a hard regression gate on a variable endpoint."""
    descriptor = endpoint.descriptor()
    if not descriptor.gate_eligible:
        raise ConfigError(
            f"a regression gate may not be built on a {descriptor.mode.value} endpoint: "
            "its output and latency are not under this benchmark's control",
            context=ErrorContext(phase=Phase.PREFLIGHT, details={"mode": descriptor.mode.value}),
        )
