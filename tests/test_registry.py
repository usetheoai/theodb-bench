

# ------------------------ B-069 bullet 2: a varredura de rerank tem de varrer
#
# O bullet pede que "o efeito do pre_reordering_num_neighbors do ScaNN seja uma varredura
# REGISTRADA, nao um script". Ele estava registrado e nao varria: o comentario acima da
# suite diz que "the rerank depth is swept together with the leaves because the two trade
# against each other", e o codigo declarava `(100,)` — um valor.
#
# Importa porque e o eixo que decide o veredito: profundidade de rerank compra recall as
# custas de QPS, e uma fronteira medida com ela FIXA nao e uma fronteira — e um ponto,
# repetido tres vezes sob rotulos diferentes de num_leaves_to_search.


def _suites_scann() -> dict:
    from theodb_bench.registry import BENCHMARKS

    return {
        nome: e
        for nome, e in BENCHMARKS.items()
        if any(i.kind == "scann" for i in getattr(e.workload, "indexes", ()) or ())
    }


def test_the_rerank_depth_effect_is_a_registered_sweep() -> None:
    """O que o B-069 bullet 2 pede: o efeito existe medido, e num registro, nao num script."""
    varrem = {
        nome: (e.workload.search_sweep or {}).get("pre_reordering_num_neighbors")
        for nome, e in _suites_scann().items()
    }
    com_varredura = {n: v for n, v in varrem.items() if v and len(v) > 1}
    assert com_varredura, (
        "nenhuma suite registrada varre pre_reordering_num_neighbors — o efeito dele so "
        f"poderia ser conhecido por script. Declarado hoje: {varrem}"
    )


def test_a_suite_that_declares_the_rerank_depth_does_not_pin_it() -> None:
    """Declarar um botao e fixa-lo num valor reporta um ponto sob rotulo de fronteira.

    NAO exige a varredura de TODA suite scann, e a diferenca importa. A primeira versao
    deste teste exigia, e reprovou `vector/synthetic/scann-sweep` — que, conferido, NAO
    mente: a descricao dela nomeia o botao que varre (`num_leaves_to_search`) e nao promete
    o outro. Estreitar o teste aqui e corrigir o teste, nao afrouxar o portao; o que ele
    passa a cobrir e o caso real — quem DECLARA a profundidade tem de varre-la.
    """
    for nome, entrada in _suites_scann().items():
        v = (entrada.workload.search_sweep or {}).get("pre_reordering_num_neighbors")
        if v is None:
            continue
        assert len(v) > 1, (
            f"{nome}: pre_reordering_num_neighbors={v} declara a profundidade e a fixa — "
            "isso e um ponto de operacao repetido sob rotulos diferentes do outro botao"
        )
