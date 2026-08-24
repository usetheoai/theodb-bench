"""`--index-repetitions` reconstrói o índice, e o bundle declara que reconstruiu.

**Por que existe.** `--repetitions` repete apenas a MEDIÇÃO: o índice era construído uma vez, antes do
laço, e toda conclusão sobre recall repousava sobre um grafo só. Medido no B-108 em 2026-08-24: o
mesmo build produz de **30 a 44** nós inalcançáveis entre corridas, e um delta de recall de +0,0015 é
indistinguível dessa variância sem reamostrar a construção. Até aqui a reamostragem se fazia
fabricando tags distintas em `ops/bench-run.sh`, o que reconstrói a imagem Docker inteira por perna.

**O que estes testes impedem.** Que a opção seja aceita e não tenha efeito — a classe
`instrumento-reporta-o-pedido`, que este projeto já pagou onze vezes num único dia. Contar
construções é a única prova; contar repetições passaria com um índice só.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from theodb_bench.adapters.base import IndexSpec
from theodb_bench.errors import ConfigError

from test_runner_e2e import _request  # a fabrica de RunRequest ja existente
from theodb_bench.runner import run_benchmark


def _bundle(tmp_path: Path, **kwargs):
    return run_benchmark(_request(tmp_path, **kwargs))


def test_uma_construcao_por_repeticao_de_indice(tmp_path: Path) -> None:
    """O número de valores de `build_seconds` É o número de construções — não o de medições."""
    payload = _bundle(tmp_path, repetitions=2, index_repetitions=3).bundle.read_artifact(
        "statistics"
    )
    vistos = 0
    for point in payload["points"]:
        m = point["metrics"].get("build_seconds")
        if m is None:
            continue
        vistos += 1
        assert m["repetitions"] == 3, f"{point['label']}: {m['repetitions']} construcoes, esperado 3"
        assert len(m["values"]) == 3
        # As medicoes seguem sendo 3 construcoes x 2 repeticoes.
        assert point["metrics"]["throughput_per_second"]["repetitions"] == 6
    assert vistos, "nenhum ponto trouxe build_seconds — o teste passaria trivialmente"


def test_o_default_nao_muda_nada(tmp_path: Path) -> None:
    """Quem não pedir reamostragem paga o custo de sempre: uma construção por ponto."""
    payload = _bundle(tmp_path, repetitions=3).bundle.read_artifact("statistics")
    for point in payload["points"]:
        m = point["metrics"].get("build_seconds")
        if m is not None:
            assert m["repetitions"] == 1


def test_o_bundle_DECLARA_se_reconstruiu(tmp_path: Path) -> None:
    """Sem a declaração, dois bundles com desvios diferentes são incomparáveis: um mede variância de
    medição e o outro de construção, e nada no artefato distingue."""
    com = _bundle(tmp_path / "com", repetitions=1, index_repetitions=2).bundle.read_artifact(
        "benchmark"
    )
    sem = _bundle(tmp_path / "sem", repetitions=1).bundle.read_artifact("benchmark")
    assert com["repetition_policy"]["rebuild_index"] is True
    assert sem["repetition_policy"]["rebuild_index"] is False
    # Os outros quatro sao verdadeiros e uteis: o arnes nao reinicia servidor nem derruba cache.
    for artefato in (com, sem):
        assert artefato["repetition_policy"]["drop_caches"] is False
        assert artefato["repetition_policy"]["restart_system"] is False


def test_zero_construcoes_e_RECUSADO(tmp_path: Path) -> None:
    """Um pedido que produziria ponto sem índice nenhum é erro de configuração, não zero silencioso."""
    from theodb_bench.bench.vector import VectorBenchmark

    with pytest.raises(ConfigError, match="index_repetitions"):
        VectorBenchmark.run_point(
            object.__new__(VectorBenchmark),
            adapter=None,  # type: ignore[arg-type]
            index=IndexSpec(kind="none"),
            search={},
            repetitions=1,
            index_repetitions=0,
        )


def test_o_artefato_result_tambem_traz_o_build_UMA_vez_por_construcao(tmp_path: Path) -> None:
    """Dois artefatos do mesmo bundle nao podem discordar.

    MEDIDO em 2026-08-24: ao mover o build para o ponto eu esqueci o `result`, que lia da repeticao.
    O `statistics` reportava build de 33,65 s e o `result` nao mencionava build nenhum — um consumidor
    que lesse o segundo concluiria que o dado nao foi coletado.
    """
    saida = _bundle(tmp_path, repetitions=3, index_repetitions=2)
    result = saida.bundle.read_artifact("result")
    stats = saida.bundle.read_artifact("statistics")

    conferidos = 0
    for ponto_r, ponto_s in zip(result["points"], stats["points"], strict=True):
        no_stats = ponto_s["metrics"].get("build_seconds")
        if no_stats is None:
            continue
        conferidos += 1
        no_result = [
            r["resources"]["build_seconds"]
            for r in ponto_r["repetitions"]
            if "build_seconds" in r.get("resources", {})
        ]
        assert len(no_result) == no_stats["repetitions"] == 2, (
            f"{ponto_r['label']}: result traz {len(no_result)} builds, statistics diz "
            f"{no_stats['repetitions']} — os dois artefatos discordam"
        )
        assert no_result == no_stats["values"]
    assert conferidos, "nenhum ponto com build — o teste passaria trivialmente"
