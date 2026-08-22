"""A linha de resumo do `run` nao pode dizer que algo nao foi medido quando foi."""

from __future__ import annotations


def test_the_summary_names_the_quality_metric_the_suite_actually_reports() -> None:
    """MEDIDO em 2026-08-22: `retrieval/synthetic/hybrid` reportou `recall=not measured`
    no terminal enquanto o bundle trazia `ndcg_at_10` de 0,0365 a 0,8266 nas tres pernas.

    A suite mede qualidade; ela so nao a chama de `recall`. Um operador lendo o terminal
    concluiria que a corrida nao mediu qualidade — e essa e a mesma classe de defeito que
    `wiki/guides/instrumento-reporta-o-pedido.md` registra: o instrumento respondeu sobre a
    metrica que ELE conhece, nao sobre a que a corrida produziu.
    """
    from theodb_bench.cli import qualidade_do_ponto

    class _M:
        def __init__(self, mediana: float) -> None:
            self.median = mediana

    assert qualidade_do_ponto({"recall": _M(0.93)}) == ("recall", 0.93)
    assert qualidade_do_ponto({"ndcg_at_10": _M(0.6225)}) == ("ndcg@10", 0.6225)
    # Preferencia estavel quando as duas existem: recall e a metrica do eixo vetorial e o
    # `run` e usado sobretudo la; a de retrieval aparece quando a outra nao existe.
    assert qualidade_do_ponto({"recall": _M(0.9), "ndcg_at_10": _M(0.6)}) == ("recall", 0.9)
    assert qualidade_do_ponto({}) is None
