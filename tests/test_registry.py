

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


# ------------------------ o nome da suite promete um dataset; o portao cobra
#
# Uma suite chamada `vector/sift1m/hnsw` rodada SEM `--dataset` mede o corpus sintetico
# semeado — a ajuda do proprio flag diz "measure a verified dataset INSTEAD of the seeded
# synthetic corpus". O bundle resultante diz `sift1m` no id e nao traz bloco `dataset`, e
# so quem abre o artefato percebe.
#
# Importa para o B-057 especificamente: descritores SIFT sao features reais de imagem, com
# anisotropia e clusterizacao; vetores sinteticos semeados nao tem essa estrutura. O
# quantizador AH que o ADR-0035 credita pelo gap de 25x e ANISOTROPICO — medi-lo em dado
# sintetico mede outra coisa e a rotula com o nome do dataset.
#
# A declaracao e um CAMPO, nao uma heuristica sobre o nome: casar `sift` por substring
# acertaria hoje e erraria no dia em que alguem registrar `vector/synthetic/sift-like`.


def test_suites_named_after_a_dataset_declare_that_they_require_it() -> None:
    from theodb_bench.registry import BENCHMARKS

    nomeadas = [n for n in BENCHMARKS if "/sift" in n or "/bigann" in n or "/scifact" in n]
    assert nomeadas, "nenhuma suite nomeada por dataset"
    sem_declaracao = [n for n in nomeadas if not BENCHMARKS[n].requires_dataset]
    assert not sem_declaracao, (
        f"suites cujo id nomeia um dataset e que nao o exigem: {sem_declaracao} — "
        "rodadas sem --dataset elas medem o corpus sintetico sob o nome do dataset"
    )


def test_a_synthetic_suite_does_not_require_a_dataset() -> None:
    """O portao nao pode virar imposto sobre quem honestamente se chama sintetico."""
    from theodb_bench.registry import BENCHMARKS

    for nome, entrada in BENCHMARKS.items():
        if "/synthetic/" in nome:
            assert entrada.requires_dataset is None, f"{nome} nao deveria exigir dataset"


def test_the_run_refuses_a_dataset_suite_without_its_dataset() -> None:
    """Declarar sem cobrar e campo decorativo — o portao de codigo morto ja cobrou isso aqui.

    Recusa em vez de aviso porque o dano e silencioso: a corrida TERMINA, o bundle sai com
    `sift1m` no id e sem bloco `dataset`, e o numero entra num conceito com o nome do
    dataset. Um aviso no stdout de uma corrida remota de 40 minutos nao e lido por ninguem.
    """
    import pytest as _pytest
    from theodb_bench.errors import ConfigError
    from theodb_bench.registry import BENCHMARKS, require_declared_dataset

    require_declared_dataset(BENCHMARKS["vector/synthetic/smoke"], None)  # nao exige: passa
    require_declared_dataset(BENCHMARKS["vector/sift1m/hnsw"], "sift-128-euclidean")  # certo

    with _pytest.raises(ConfigError) as sem:
        require_declared_dataset(BENCHMARKS["vector/sift1m/hnsw"], None)
    assert "sift-128-euclidean" in str(sem.value)

    with _pytest.raises(ConfigError) as outro:
        require_declared_dataset(BENCHMARKS["vector/sift1m/hnsw"], "beir-scifact")
    assert "beir-scifact" in str(outro.value)
