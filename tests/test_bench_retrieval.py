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


# ------------------------ o corpus sintetico perdia o sinal lexical com a escala
#
# O gerador declara a intencao: "the query vector sits near the primary document, with
# noise, so the dense leg and the lexical leg AGREE ON THE PRIMARY and disagree elsewhere
# -- which is the situation fusion exists for."
#
# MEDIDO em 2026-08-22: essa intencao vale a corpus pequeno e QUEBRA com a escala, porque o
# vocabulario era fixo em 28 termos. Documentos que contem a consulta INTEIRA, em media:
#
#     corpus=  200 ->  1,6      corpus= 1000 ->  3,9
#     corpus=  500 ->  3,3      corpus= 5000 -> 38,9
#
# A 5.000 documentos o BM25 enfrenta ~39 candidatos igualmente casados para 1 relevante, e
# a perna lexical marcou nDCG@10 = 0,0752 contra 0,8266 da vetorial. Nao e defeito do motor
# lexical: e o corpus nao ter sinal. E a fusao, fundindo uma perna forte com ruido, PIOROU
# a perna forte — o que e o comportamento correto do RRF sobre uma pergunta que ninguem
# deveria fazer.


def _casam_a_consulta_inteira(corpus_size: int, query_count: int = 40) -> float:
    from theodb_bench.bench.retrieval import RetrievalWorkload, generate_corpus

    docs, qs = generate_corpus(RetrievalWorkload(corpus_size=corpus_size, query_count=query_count))
    conjuntos = [set(d.text.split()) for d in docs]
    total = 0
    for i in range(query_count):
        termos = set(qs.texts[i].split())
        total += sum(1 for c in conjuntos if termos <= c)
    return total / query_count


def test_the_lexical_leg_keeps_signal_as_the_corpus_grows() -> None:
    """A discriminacao lexical nao pode evaporar quando alguem sobe o `corpus_size`.

    O limite de 8 e generoso de proposito: com ~1 relevante por consulta, ate um punhado de
    candidatos ainda deixa o BM25 ordenar. O que se recusa e a ordem de grandeza — 39
    candidatos empatados nao e ranking, e sorteio.
    """
    for n in (200, 1000, 5000, 20000):
        media = _casam_a_consulta_inteira(n)
        assert media <= 8.0, (
            f"corpus={n}: {media:.1f} documentos casam a consulta inteira, em media. "
            "A perna lexical nao tem o que ordenar, e qualquer numero dela mede o corpus, "
            "nao o motor."
        )


# ------------------------ B-005: o teste pareado de QUALIDADE precisa do dado por consulta
#
# O `dod` do B-005 pede "uma fusao cujo ganho sobre o vetorial puro sobreviva a teste
# pareado de significancia". MEDIDO em 2026-08-22: o nDCG e IDENTICO nas cinco repeticoes
# (0,8266 cinco vezes) porque o corpus e as consultas sao deterministicos — qualidade nao
# varia entre repeticoes, so a vazao varia. Um teste pareado ENTRE REPETICOES tem variancia
# zero por construcao e nao diz nada.
#
# A unidade certa e a CONSULTA: 300 pares (fusao, vetorial) sobre as mesmas consultas. O
# arnes guardava `latency_by_query` e descartava o nDCG por consulta, entao o teste que o
# item pede era impossivel de fazer com o que o bundle trazia.
#
# E `compare.py` ja tem `pair_by_query` e `render_paired_verdict`, usados para latencia.
# Faltava o dado.


def test_the_quality_is_kept_per_query_not_only_averaged() -> None:
    from theodb_bench.adapters.fake import FakeAdapter
    from theodb_bench.bench.retrieval import RetrievalBenchmark, RetrievalWorkload, generate_corpus

    w = RetrievalWorkload(corpus_size=300, query_count=25, pipelines=("vector",))
    docs, qs = generate_corpus(w)
    adapter = FakeAdapter()
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    bench = RetrievalBenchmark(w, documents=docs, queries=qs)
    bench.load(adapter)
    resultado = bench.run_pipeline(adapter, "vector", 1)

    assert resultado.quality_by_query, "o nDCG por consulta nao e guardado"
    assert len(resultado.quality_by_query) == 25
    assert all(0.0 <= v <= 1.0 for v in resultado.quality_by_query.values())
    # A media tem de ser a media do que foi guardado — se divergirem, uma das duas mente.
    media = sum(resultado.quality_by_query.values()) / len(resultado.quality_by_query)
    assert abs(media - (resultado.ndcg_at_10 or 0.0)) < 1e-9


def test_the_fusion_is_compared_to_the_vector_leg_by_paired_query() -> None:
    """O que o B-005 pede, e agora e possivel: 300 pares (fusao, vetorial) nas MESMAS consultas."""
    from theodb_bench.bench.retrieval import veredito_de_qualidade

    # Fusao ganha em 3 de 4 consultas, e a diferenca e grande.
    ganha = veredito_de_qualidade(
        "hybrid_rrf", {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.2},
        "vector", {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.6},
    )
    assert ganha and "hybrid_rrf" in ganha

    # Sem sobreposicao de consultas nao ha par: a resposta honesta e ausencia, nao zero.
    assert veredito_de_qualidade("a", {0: 0.9}, "b", {5: 0.5}) is None
    assert veredito_de_qualidade("a", {}, "b", {}) is None


def test_the_per_query_quality_reaches_the_repetition_the_runner_emits() -> None:
    """Sem isto o dado existe no PipelineResult e morre la — B-005 continua sem resposta.

    O runner emite `raw/latency-by-query.json` a partir de `RepetitionResult.latency_by_query`,
    e o docstring dele diz por que: "summaries cannot be paired, and this is the ONLY place
    the per-query values exist". O mesmo argumento vale para qualidade, e a unidade do teste
    pareado de qualidade e a consulta — nao a repeticao, cujo nDCG e identico nas cinco.
    """
    from theodb_bench.bench.retrieval import RetrievalBenchmark, RetrievalWorkload

    w = RetrievalWorkload(corpus_size=200, query_count=12, pipelines=("vector",))
    bench = RetrievalBenchmark(w)
    from theodb_bench.adapters.fake import FakeAdapter

    adapter = FakeAdapter()
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    bench.load(adapter)
    pontos = bench.points(adapter, repetitions=1)

    assert pontos, "nenhum ponto"
    for ponto in pontos:
        for rep in ponto.repetitions:
            assert rep.quality_by_query, (
                f"{ponto.label}: a qualidade por consulta nao chega a repeticao, entao nao "
                "chega ao artefato e o teste pareado continua impossivel"
            )
            assert len(rep.quality_by_query) == 12


def test_the_quality_verdict_knows_higher_is_better() -> None:
    """MEDIDO em 2026-08-22, e por pouco nao virou publicacao: o veredito dizia
    'hybrid_rrf BEATS vector' sobre `mean diff = -0.007`, com a fusao em 0,8195 e a
    vetorial em 0,8266. A fusao e PIOR.

    Causa: `render_paired_verdict` nasceu para LATENCIA, onde menor e melhor, e tem
    `lower_is_better=True` por default. Para nDCG maior e melhor, e eu nao passei o
    parametro — que ja existia. O texto ate imprimia 'faster', que nao significa nada
    para qualidade.

    Este teste existe porque a inversao e invisivel na leitura: 'A beats B' parece certo
    ate alguem conferir o sinal.
    """
    from theodb_bench.bench.retrieval import veredito_de_qualidade

    # Amostra grande o bastante para o teste ter poder — com quatro consultas nao ha
    # significancia, e o veredito sai "indistinguishable", onde a ordem dos nomes e a dos
    # argumentos e nao a do vencedor. A primeira versao deste teste caiu nisso.
    melhor = {i: 0.80 + (i % 5) * 0.01 for i in range(120)}
    pior = {i: 0.60 + (i % 5) * 0.01 for i in range(120)}

    v = veredito_de_qualidade("pior", pior, "melhor", melhor)
    assert v, "sem veredito"
    assert "**melhor** beats **pior**" in v, (
        f"o vencedor tem de ser `melhor`, que tem nDCG maior em toda consulta: {v}"
    )
    assert "faster" not in v, f"'faster' nao significa nada para qualidade: {v}"

    # E o inverso, para o teste nao passar por acidente de ordem dos argumentos.
    inverso = veredito_de_qualidade("melhor", melhor, "pior", pior)
    assert inverso and "**melhor** beats **pior**" in inverso, inverso
