"""Four pipelines, one corpus, one query set. Quality and speed are both axes."""

from __future__ import annotations

import pytest
from theodb_bench.adapters.fake import FakeAdapter, FakeConfig
from theodb_bench.bench.retrieval import (
    HYBRID_RRF,
    HYBRID_RRF_RERANK,
    LEXICAL,
    PIPELINES,
    VECTOR,
    RetrievalBenchmark,
    RetrievalWorkload,
    generate_corpus,
)
from theodb_bench.errors import ConfigError


def _workload(**overrides: object) -> RetrievalWorkload:
    base: dict[str, object] = {"corpus_size": 200, "query_count": 30, "dimension": 32, "n": 20}
    base.update(overrides)
    return RetrievalWorkload(**base)  # type: ignore[arg-type]


def _ready(config: FakeConfig | None = None) -> FakeAdapter:
    adapter = FakeAdapter(config)
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    return adapter


def _loaded(workload: RetrievalWorkload) -> tuple[FakeAdapter, RetrievalBenchmark]:
    adapter = _ready()
    benchmark = RetrievalBenchmark(workload)
    benchmark.load(adapter)
    return adapter, benchmark


# ------------------------------------------------------------------- workload


def test_unknown_pipelines_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown pipeline"):
        _workload(pipelines=("telepathy",))


def test_k_beyond_the_candidate_depth_is_refused() -> None:
    # Retrieving 10 candidates and scoring recall@50 would score positions that
    # were never retrieved.
    with pytest.raises(ConfigError, match="exceeds the candidate depth"):
        _workload(k=50, n=10)


# --------------------------------------------------------------------- corpus


def test_the_corpus_is_deterministic_for_a_seed() -> None:
    first, first_queries = generate_corpus(_workload(seed=7))
    second, second_queries = generate_corpus(_workload(seed=7))
    assert [d.text for d in first] == [d.text for d in second]
    assert first_queries.texts == second_queries.texts


def test_a_different_seed_gives_a_different_corpus() -> None:
    first, _ = generate_corpus(_workload(seed=7))
    second, _ = generate_corpus(_workload(seed=8))
    assert [d.text for d in first] != [d.text for d in second]


def test_every_query_has_judgements() -> None:
    _, queries = generate_corpus(_workload())
    assert len(queries.relevance) == len(queries.texts)
    assert all(judgements for judgements in queries.relevance)


def test_judgements_are_graded_not_binary() -> None:
    # nDCG needs grades; a binary set would make it equivalent to recall.
    _, queries = generate_corpus(_workload())
    grades = {gain for judgements in queries.relevance for gain in judgements.values()}
    assert len(grades) > 1


def test_every_relevant_document_exists_in_the_corpus() -> None:
    documents, queries = generate_corpus(_workload())
    ids = {document.id for document in documents}
    for judgements in queries.relevance:
        assert set(judgements) <= ids


# ------------------------------------------------------------------ pipelines


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_every_pipeline_runs_and_reports_both_axes(pipeline: str) -> None:
    workload = _workload(pipelines=(pipeline,))
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run_pipeline(adapter, pipeline, repetition=1)
        assert result.status == "measured"
        assert result.successes == workload.query_count
        # Quality and performance are separate axes, and both must be present.
        assert result.ndcg_at_10 is not None
        assert result.throughput is not None
    finally:
        adapter.stop()


def test_all_pipelines_share_one_corpus_and_query_set() -> None:
    # Comparing pipelines evaluated on different data measures the data.
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        texts_before = benchmark.queries.texts
        for pipeline in PIPELINES:
            benchmark.run_pipeline(adapter, pipeline, repetition=1)
        assert benchmark.queries.texts == texts_before
    finally:
        adapter.stop()


def test_lexical_retrieval_finds_the_documents_the_query_was_built_from() -> None:
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run_pipeline(adapter, LEXICAL, repetition=1)
        assert result.recall_at_k is not None
        assert result.recall_at_k > 0.0
    finally:
        adapter.stop()


def test_hybrid_reports_a_stage_breakdown() -> None:
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run_pipeline(adapter, HYBRID_RRF, repetition=1)
        assert {"lexical", "vector", "fusion"} <= set(result.stage_seconds)
    finally:
        adapter.stop()


def test_rerank_charges_model_time_to_its_own_stage() -> None:
    # The database's contribution must stay visible; a composite that hides
    # inference measures the model vendor.
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run_pipeline(adapter, HYBRID_RRF_RERANK, repetition=1)
        assert "rerank_model" in result.stage_seconds
        assert "rerank_database" in result.stage_seconds
        assert result.stage_seconds["rerank_model"] > 0
    finally:
        adapter.stop()


def test_an_unsupported_pipeline_is_reported_not_failed() -> None:
    workload = _workload()
    adapter = _ready(FakeConfig(capabilities={"vector_exact": True, "lexical": False}))
    benchmark = RetrievalBenchmark(workload)
    benchmark.load(adapter)
    try:
        result = benchmark.run_pipeline(adapter, LEXICAL, repetition=1)
        assert result.status == "unsupported"
        assert result.successes == 0
        assert result.ndcg_at_10 is None
    finally:
        adapter.stop()


def test_quality_is_absent_rather_than_zero_when_nothing_was_measured() -> None:
    workload = _workload()
    adapter = _ready(FakeConfig(capabilities={"vector_exact": True}))
    benchmark = RetrievalBenchmark(workload)
    benchmark.load(adapter)
    try:
        result = benchmark.run_pipeline(adapter, HYBRID_RRF, repetition=1)
        assert result.ndcg_at_10 is None
    finally:
        adapter.stop()


# --------------------------------------------------------------- offline twin


def test_the_offline_fusion_matches_the_systems_own() -> None:
    # Having both is the point: a divergence is a finding, not a mystery.
    from theodb_bench.adapters.base import HybridQuery, KnnQuery, LexicalQuery

    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        text = benchmark.queries.texts[0]
        vector = benchmark.queries.vectors[0]
        lexical = adapter.execute_lexical(LexicalQuery(workload.table, text, workload.n))
        dense = adapter.execute(
            KnnQuery(table=workload.table, vector=vector, k=workload.n, metric=workload.metric)
        )
        system = adapter.execute_hybrid(
            HybridQuery(
                table=workload.table,
                text=text,
                vector=vector,
                n=workload.n,
                metric=workload.metric,
            )
        )
        assert list(system.ids) == benchmark.offline_fusion(lexical.ids, dense.ids)
    finally:
        adapter.stop()


# ------------------------------------------------------------------- summary


def test_the_summary_compares_pipelines_side_by_side() -> None:
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        results = [benchmark.run_pipeline(adapter, p, 1) for p in PIPELINES]
        summary = benchmark.summary(results)
        assert set(summary["pipelines"]) == set(PIPELINES)
        for entry in summary["pipelines"].values():
            assert "ndcg_at_10" in entry
            assert "throughput_per_second" in entry
    finally:
        adapter.stop()


def test_metric_series_expose_quality_and_latency_together() -> None:
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        series = benchmark.run_pipeline(adapter, VECTOR, 1).metric_series()
        assert "ndcg_at_10" in series
        assert "throughput_per_second" in series
        assert any(name.startswith("latency_") for name in series)
    finally:
        adapter.stop()
