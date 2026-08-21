"""B-093 — o corpus com julgamento humano que o arnes nao tinha.

O `retrieval.py` traz nDCG@10, Recall@k, MRR e quatro pipelines, e NENHUM benchmark registrado o
usava: o unico corpus era o `generate_corpus` sintetico, cuja docstring diz que exercita a pipeline
sem ser alegacao de qualidade. Todo numero lexical publicado saiu de script ad-hoc.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from theodb_bench.bench.beir import load_beir
from theodb_bench.errors import DatasetError


def _monta(raiz: Path, *, qrels: str, corpus_ids: tuple[str, ...] = ("d1", "d2")) -> Path:
    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / "qrels").mkdir(exist_ok=True)
    with (raiz / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for i, did in enumerate(corpus_ids):
            fh.write(json.dumps({"_id": did, "title": f"t{i}", "text": f"corpo {i}"}) + "\n")
    with (raiz / "queries.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_id": "q1", "text": "consulta um"}) + "\n")
        fh.write(json.dumps({"_id": "q2", "text": "consulta dois"}) + "\n")
    (raiz / "qrels" / "test.tsv").write_text(qrels, encoding="utf-8")
    return raiz


def test_reads_a_directory(tmp_path: Path) -> None:
    raiz = _monta(tmp_path / "c", qrels="query-id\tcorpus-id\tscore\nq1\td2\t1\n")
    docs, consultas = load_beir(raiz)
    assert [d.text for d in docs] == ["t0 corpo 0", "t1 corpo 1"]
    assert consultas.texts == ("consulta um",)
    assert consultas.relevant_ids(0) == {1}, "o id externo `d2` mapeia para o indice interno 1"


def test_a_zip_and_a_directory_give_the_same_thing(tmp_path: Path) -> None:
    """O zip publicado e o diretorio extraido nao podem divergir — senao o sha256 verifica um e o
    benchmark mede o outro."""
    raiz = _monta(tmp_path / "c", qrels="query-id\tcorpus-id\tscore\nq1\td2\t1\n")
    z = tmp_path / "c.zip"
    with zipfile.ZipFile(z, "w") as arq:
        for p in sorted(raiz.rglob("*")):
            if p.is_file():
                arq.write(p, f"scifact/{p.relative_to(raiz)}")
    d1, q1 = load_beir(raiz)
    d2, q2 = load_beir(z)
    assert [d.text for d in d1] == [d.text for d in d2]
    assert q1.texts == q2.texts and q1.relevance == q2.relevance


def test_only_judged_queries_are_returned(tmp_path: Path) -> None:
    """Uma consulta sem qrel nao tem verdade contra a qual pontuar.

    Inclui-la faria o nDCG medio cair por uma razao que NAO e qualidade de busca — o denominador
    cresceria com zeros que so dizem que ninguem julgou aquela consulta.
    """
    raiz = _monta(tmp_path / "c", qrels="query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    _, consultas = load_beir(raiz)
    assert consultas.texts == ("consulta um",), "q2 nao tem julgamento e fica de fora"


def test_a_qrel_pointing_outside_the_corpus_is_refused(tmp_path: Path) -> None:
    """Ignorar em silencio inflaria o denominador do recall sem dizer por que."""
    raiz = _monta(tmp_path / "c", qrels="query-id\tcorpus-id\tscore\nq1\tinexistente\t1\n")
    with pytest.raises(DatasetError, match="ausente do corpus"):
        load_beir(raiz)


def test_a_judged_query_without_text_is_refused(tmp_path: Path) -> None:
    raiz = _monta(tmp_path / "c", qrels="query-id\tcorpus-id\tscore\nq9\td1\t1\n")
    with pytest.raises(DatasetError, match="sem texto"):
        load_beir(raiz)


def test_qrels_without_a_header_still_parse(tmp_path: Path) -> None:
    """Nem toda publicacao do BEIR traz cabecalho; tratar a primeira linha como dado a perderia."""
    raiz = _monta(tmp_path / "c", qrels="q1\td1\t1\n")
    _, consultas = load_beir(raiz)
    assert consultas.relevant_ids(0) == {0}


def test_the_vectors_carry_no_information_and_that_is_the_point(tmp_path: Path) -> None:
    """O BEIR nao publica embeddings. Preencher com ruido faria a perna densa RODAR e parecer
    medida.

    Este teste existe para que a ausencia seja uma decisao VERIFICADA, e nao um detalhe que alguem
    'conserta' preenchendo com `standard_normal` — que produziria um nDCG denso sem significado
    nenhum, com a mesma aparencia de um resultado.

    A asserção é sobre INFORMACAO, nao sobre largura. A primeira versao fixava `size == 0`, e
    largura zero e ilegal no PostgreSQL (`dimensions for type vector must be at least 1`) — a
    coluna existe porque a forma da tabela a exige. O que nao pode mudar e ela nao dizer nada.
    """
    raiz = _monta(tmp_path / "c", qrels="query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    docs, consultas = load_beir(raiz)
    assert all(not d.vector.any() for d in docs), "um vetor com valor seria um embedding inventado"
    assert not consultas.vectors.any()
    assert all(d.vector.size <= 1 for d in docs), "tao estreito quanto o PostgreSQL permite"
