"""The capability matrix, generated rather than written.

B-073 asks for a report that lists every declared capability and which adapters
exercise it, and asks for it to be *generated* — because the version that gets
written by hand is the version that drifts. The measurement that opened the item
was exactly this crossing done manually: six of fourteen capabilities reachable,
eight with no adapter at all. A month later nobody would know which number is
current, and the README would still say six.

Generated has a second property that matters more here: a capability an adapter
*declares* but cannot reach is invisible to a hand-written table and visible to
this one, because both sides come from the same registry the runs use.
"""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import CAPABILITIES
from theodb_bench.capabilities import capability_matrix, render_capability_matrix


def test_every_declared_capability_appears_exactly_once() -> None:
    matrix = capability_matrix()

    assert [row.capability for row in matrix] == list(CAPABILITIES)


def test_a_capability_names_the_adapters_that_declare_it() -> None:
    matrix = {row.capability: row for row in capability_matrix()}

    assert "theodb" in matrix["vector_hnsw"].adapters
    assert "pgvector" in matrix["vector_hnsw"].adapters


def test_a_capability_nobody_declares_is_reported_as_such_not_omitted() -> None:
    """An absent row reads as "not a thing"; an empty row reads as "nothing
    reaches this yet", which is the true and useful statement."""
    matrix = {row.capability: row for row in capability_matrix()}

    for capability in ("rerank", "vectorizer", "ai_sql"):
        assert matrix[capability].adapters == ()
        assert not matrix[capability].reachable


def test_the_fake_adapter_does_not_count_as_reach() -> None:
    """The fake exists to test the harness, and counting it would let the matrix
    report a pillar as reachable when no real system implements it."""
    matrix = {row.capability: row for row in capability_matrix()}

    assert all("fake" not in row.adapters for row in matrix.values())


def test_the_summary_counts_what_is_reachable() -> None:
    matrix = capability_matrix()

    reachable = sum(1 for row in matrix if row.reachable)

    assert reachable == 11, f"expected 11 of {len(CAPABILITIES)} reachable, got {reachable}"


def test_the_rendered_table_names_every_capability() -> None:
    rendered = render_capability_matrix()

    for capability in CAPABILITIES:
        assert capability in rendered


def test_the_rendered_table_carries_the_count_it_is_read_for() -> None:
    rendered = render_capability_matrix()

    assert f"11 of {len(CAPABILITIES)}" in rendered


def test_the_readme_table_is_the_generated_one() -> None:
    """The check that makes this worth generating: the README's table must be
    what the registry says today, not what it said when someone typed it."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    rendered = render_capability_matrix()

    assert rendered.strip() in readme, (
        "the README's capability matrix is stale; regenerate it with "
        "`theodb-bench capabilities --markdown`"
    )


def test_the_cli_prints_the_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    from theodb_bench.cli import main

    assert main(["capabilities"]) == 0
    assert "11 of 14" in capsys.readouterr().out


# ------------------------ B-104: declarada nao e medida, e a matriz nao distinguia
#
# A matriz respondia "quais adapters DECLARAM X". Nao respondia "quantas suites MEDEM X",
# e a diferenca e a distincao central deste projeto. Medido em 2026-08-22: `hybrid` aparece
# declarada pelo `theodb` — e nenhuma das 23 suites registradas a exercita. O
# `retrieval/scifact/lexical` ate explica por que fica de fora ("BEIR publishes no
# embeddings"), o que esta certo e deixa o pilar sem instrumento.
#
# Importa porque o ADR-0033, assinado, poe a diferenciacao AI-native como meta: o eixo em
# que se pretende GANHAR tem zero suites, e o eixo em que a meta e EMPATAR tem 18.


def test_the_matrix_says_how_many_suites_measure_each_capability() -> None:
    from theodb_bench.capabilities import capability_matrix

    linhas = {r.capability: r for r in capability_matrix()}
    assert hasattr(next(iter(linhas.values())), "suites"), "a matriz nao reporta suites"
    # O vetorial e o eixo mais medido do projeto — se ele vier zero, a derivacao esta errada.
    assert linhas["vector_hnsw"].suites, "vector_hnsw sem suite significa derivacao quebrada"


def test_a_capability_declared_without_a_suite_is_visible_as_such() -> None:
    """O achado que motivou isto: capacidade declarada e nao medida tem de APARECER."""
    from theodb_bench.capabilities import capability_matrix, render_capability_matrix

    linhas = {r.capability: r for r in capability_matrix()}
    declaradas_sem_suite = [c for c, r in linhas.items() if r.adapters and not r.suites]
    # `hybrid` estava nesta lista quando o teste foi escrito, e SAIU quando a suite
    # `retrieval/synthetic/hybrid` foi registrada no mesmo dia. O assert que a nomeava
    # cumpriu seu papel e foi removido; o que fica e a propriedade GERAL, que continua
    # valendo para a proxima capacidade que alguem declarar sem medir.
    texto = render_capability_matrix()
    if declaradas_sem_suite:
        assert "declarada e não medida" in texto.lower(), (
            f"ha capacidade declarada e nao medida ({declaradas_sem_suite}) e o texto nao "
            "diz isso — a coluna sozinha nao e lida"
        )
    else:
        assert "declarada e não medida" not in texto.lower(), (
            "nenhuma capacidade esta declarada-sem-suite e o texto ainda avisa que ha"
        )


def test_the_hybrid_capability_has_a_suite_measuring_it() -> None:
    """B-104/B-005: o pilar que a meta assinada chama de diferenciacao tem de ser medivel.

    A maquinaria ja existia inteira — `retrieval.py` declara as pernas lexical, vector,
    hybrid_rrf e hybrid_rrf_rerank, o adapter implementa `execute_hybrid` chamando
    `ai.hybrid_search_rrf`, e o gerador sintetico produz corpus com texto E vetores com
    ground truth construido. O que faltava era UMA entrada de registro.
    """
    from theodb_bench.capabilities import capability_matrix

    linha = {r.capability: r for r in capability_matrix()}["hybrid"]
    assert linha.suites, "nenhuma suite registrada mede `hybrid`"
    assert not linha.declared_unmeasured


def test_the_derivation_uses_the_declared_pipelines_not_the_suite_name() -> None:
    """Nome de suite e convencao; `pipelines` e declaracao. A segunda nao mente.

    `retrieval.py` ja mapeia perna->capacidade em PIPELINE_CAPABILITY, e derivar dali e
    exato: uma suite que declare so a perna lexical NAO deve contar como medindo `hybrid`
    so porque o id dela comeca com `retrieval/`.
    """
    from theodb_bench.capabilities import suites_by_capability
    from theodb_bench.registry import BENCHMARKS

    medindo = suites_by_capability()
    for suite in medindo.get("hybrid", ()):
        pernas = getattr(BENCHMARKS[suite].workload, "pipelines", ())
        assert "hybrid_rrf" in pernas, f"{suite} conta como hibrida sem declarar a perna"
