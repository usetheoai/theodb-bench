"""External inference is a reproducibility risk; these tests pin how it is handled."""

from __future__ import annotations

import numpy as np
import pytest
from theodb_bench.ai import (
    CostBreakdown,
    EndpointMode,
    LocalEndpoint,
    MockEndpoint,
    RemoteEndpoint,
    require_gate_eligible,
)
from theodb_bench.errors import ConfigError

# ----------------------------------------------------------------------- mock


def test_the_mock_is_deterministic() -> None:
    endpoint = MockEndpoint()
    assert endpoint.generate("same prompt").output == endpoint.generate("same prompt").output


def test_different_prompts_give_different_output() -> None:
    endpoint = MockEndpoint()
    assert endpoint.generate("a").output != endpoint.generate("b").output


def test_the_mock_latency_must_be_non_zero() -> None:
    # A model that answers instantly changes the concurrency regime of the
    # whole loop and flatters any system that overlaps I/O with inference.
    with pytest.raises(ConfigError, match="non-zero"):
        MockEndpoint(latency_seconds=0.0)


def test_the_mock_actually_costs_the_declared_time() -> None:
    call = MockEndpoint(latency_seconds=0.02).generate("prompt")
    assert call.inference_seconds >= 0.02


def test_the_mock_embeds_deterministically_and_normalised() -> None:
    endpoint = MockEndpoint()
    first, _ = endpoint.embed("hello", 16)
    second, _ = endpoint.embed("hello", 16)
    assert np.array_equal(first, second)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-6)


def test_the_mock_embedding_carries_no_semantics() -> None:
    # Deliberate: the mock measures cost, never semantic quality. Two similar
    # texts must not embed similarly, so nobody reads meaning into it.
    endpoint = MockEndpoint()
    first, _ = endpoint.embed("the cat sat", 32)
    second, _ = endpoint.embed("the cat sat down", 32)
    assert abs(float(np.dot(first, second))) < 0.9


def test_batching_amortises_by_the_declared_factor() -> None:
    endpoint = MockEndpoint(latency_seconds=0.01, batch_efficiency=0.1)
    calls = endpoint.generate_batch(["a", "b", "c", "d"])
    assert len(calls) == 4
    batched_total = sum(call.inference_seconds for call in calls)
    # Four separate calls would cost about 0.04s; batching at 0.1 efficiency
    # should cost materially less.
    assert batched_total < 0.04


def test_an_invalid_batch_efficiency_is_refused() -> None:
    with pytest.raises(ConfigError, match="batch efficiency"):
        MockEndpoint(batch_efficiency=0.0)


def test_an_empty_batch_costs_nothing() -> None:
    assert MockEndpoint().generate_batch([]) == []


# ------------------------------------------------------------------ standings


def test_only_the_mock_may_back_a_regression_gate() -> None:
    require_gate_eligible(MockEndpoint())

    with pytest.raises(ConfigError, match="may not be built on a local endpoint"):
        require_gate_eligible(LocalEndpoint("frozen-7b", lambda p: "out"))

    with pytest.raises(ConfigError, match="may not be built on a remote endpoint"):
        require_gate_eligible(RemoteEndpoint("hosted-model", lambda p: "out"))


def test_remote_is_always_marked_environment_dependent() -> None:
    descriptor = RemoteEndpoint("hosted-model", lambda p: "out").descriptor()
    assert descriptor.environment_dependent is True
    assert descriptor.gate_eligible is False


def test_local_is_reproducible_but_still_not_gate_eligible() -> None:
    descriptor = LocalEndpoint("frozen-7b", lambda p: "out").descriptor()
    assert descriptor.environment_dependent is False
    assert descriptor.gate_eligible is False


def test_an_endpoint_must_name_its_model() -> None:
    # "We used an LLM" is not a measurement condition.
    with pytest.raises(ConfigError, match="must name the model"):
        LocalEndpoint("", lambda p: "out")
    with pytest.raises(ConfigError, match="must record which model"):
        RemoteEndpoint("", lambda p: "out")


def test_descriptors_record_what_a_reader_needs_to_reproduce() -> None:
    payload = MockEndpoint(batch_size=8).descriptor().as_dict()
    assert payload["mode"] == EndpointMode.MOCK.value
    assert payload["model"]
    assert payload["batch_size"] == 8
    assert "network_placement" in payload


# ---------------------------------------------------------------- attribution


def test_the_default_batch_does_not_pretend_to_batch() -> None:
    # An endpoint that loops must not be credited with batching efficiency.
    calls = LocalEndpoint("frozen-7b", lambda p: p.upper()).generate_batch(["a", "b"])
    assert [call.output for call in calls] == ["A", "B"]


def test_remote_attributes_time_to_the_network_rather_than_inventing_a_split() -> None:
    # The client cannot see where a hosted call spent its time, so it does not
    # guess.
    call = RemoteEndpoint("hosted", lambda p: "answer").generate("prompt")
    assert call.inference_seconds == 0.0
    assert call.network_seconds > 0.0


def test_the_breakdown_separates_database_from_model_time() -> None:
    breakdown = CostBreakdown(database_seconds=0.030)
    endpoint = MockEndpoint(latency_seconds=0.01)
    breakdown.record(endpoint.generate("one"))
    breakdown.record(endpoint.generate("two"))

    assert breakdown.calls == 2
    assert breakdown.inference_seconds > 0
    assert breakdown.total_seconds > breakdown.database_seconds
    share = breakdown.database_share
    assert share is not None and 0 < share < 1


def test_an_empty_breakdown_has_no_share_rather_than_zero() -> None:
    # Zero would read as "the database contributed nothing", which is a claim.
    assert CostBreakdown().database_share is None


def test_the_breakdown_counts_tokens_for_cost_per_step() -> None:
    breakdown = CostBreakdown()
    breakdown.record(MockEndpoint().generate("three word prompt"))
    payload = breakdown.as_dict()
    assert payload["prompt_tokens"] == 3
    assert payload["completion_tokens"] > 0
