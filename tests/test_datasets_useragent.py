"""B-069/B-018 — o arnes precisa conseguir buscar o dataset que ele mesmo declara.

MEDIDO em 2026-08-21 num host limpo (droplet `g-16vcpu-64gb`, nyc3): a origem declarada do
`sift-128-euclidean` responde **403 Forbidden** ao `User-Agent` default do `urllib`
(`Python-urllib/3.12`) e **200** ao do `curl`. O CDN filtra por agente.

O defeito so nao aparecia porque toda corrida anterior encontrou o arquivo ja em disco — que e
exatamente o modo de falha que so um host limpo revela.
"""

from __future__ import annotations

import urllib.request

from theodb_bench.datasets import _request


def test_the_request_declares_who_is_downloading() -> None:
    req = _request("https://example.invalid/corpus.hdf5")
    agente = req.get_header("User-agent")
    assert agente is not None, "sem User-Agent o CDN da origem responde 403 e o fetch falha"
    assert "theodb-bench" in agente


def test_the_agent_does_not_impersonate_a_browser() -> None:
    """Passar por Mozilla funcionaria e seria mentir para o servidor sobre quem esta do outro lado.

    Um arnes cuja premissa e medir honestamente nao comeca mentindo na primeira requisicao. Este
    teste existe para que essa escolha seja uma decisao registrada, e nao um detalhe que a proxima
    pessoa com pressa desfaz.
    """
    agente = _request("https://example.invalid/corpus.hdf5").get_header("User-agent")
    assert agente is not None
    for disfarce in ("Mozilla", "Chrome", "Safari", "AppleWebKit", "Gecko"):
        assert disfarce not in agente, f"o agente nao deve fingir ser um navegador ({disfarce})"


def test_the_default_urllib_agent_is_what_the_origin_rejects() -> None:
    """Fixa a razao: o default e `Python-urllib/...`, e e ele que a origem recusa com 403."""
    padrao = urllib.request.Request("https://example.invalid/corpus.hdf5")
    assert padrao.get_header("User-agent") is None, (
        "quando o urllib nao carrega header proprio, ele emite `Python-urllib/X.Y` na conexao — "
        "que e o agente medido como recusado pela origem do sift-128-euclidean"
    )
