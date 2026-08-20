"""B-087 — a versao que o pacote reporta e a tag da release nao podem divergir.

Medido em 2026-08-20 ao cortar a `v0.2.0`: `pyproject.toml` e `src/__init__.py` declaravam ambos
`0.1.0.dev0`, e nenhum dos dois foi tocado no corte da `v0.1.0`. O `release.yml` deriva a versao da
TAG e as notas do CHANGELOG, e nunca le nenhum dos dois.

Nao e cosmetico. `environment.py:354` grava `f"theodb-bench {__version__}"` na captura de ambiente,
que vai para dentro do bundle — de modo que **todo bundle publicado registra a versao errada do
arnes**. Um bundle existe para ser reproduzivel por terceiros, e a versao da ferramenta que o
produziu e parte do que torna isso possivel.

Enquanto era `0.1.0.dev0` vs `v0.1.0` dava para ler como sufixo de desenvolvimento. `0.1.0.dev0`
vs `v0.2.0` e outra versao.
"""

from __future__ import annotations

import re
from importlib.metadata import version as metadata_version
from pathlib import Path

import theodb_bench

RAIZ = Path(__file__).resolve().parent.parent

#: `tomllib` so existe no 3.11+ e o projeto mira 3.10. A linha `version = "..."` do bloco
#: `[project]` e a primeira do arquivo com essa forma, e le-la por regex e honesto para UMA
#: asserção — nao para parsear TOML em geral.
_VERSAO_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def _versao_do_pyproject() -> str:
    m = _VERSAO_RE.search((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    assert m, 'pyproject.toml nao declara `version = "..."`'
    return m.group(1)


def test_the_package_reports_the_version_pyproject_declares() -> None:
    """UMA fonte de verdade. Duas copias divergem — e estas duas ja divergiam do release."""
    assert theodb_bench.__version__ == _versao_do_pyproject()


def test_the_version_is_not_hardcoded_a_second_time() -> None:
    """A prova ESTRUTURAL de fonte unica, e nao so de valores iguais hoje.

    `src/__init__.py` declarava o literal ao lado do `pyproject.toml`. Dois literais iguais hoje
    sao duas copias que divergem amanha — e foi assim que as duas envelheceram juntas por dois
    cortes de release sem que nada notasse.
    """
    fonte = (RAIZ / "src" / "__init__.py").read_text(encoding="utf-8")
    assert not re.search(r'__version__\s*=\s*"\d', fonte), (
        "`__version__` esta hardcoded em src/__init__.py. Derive de `importlib.metadata`, que le "
        "a mesma fonte que o pip instalou."
    )
    assert theodb_bench.__version__ == metadata_version("theodb-bench")


def test_the_declared_version_is_not_a_development_placeholder() -> None:
    """`0.1.0.dev0` sobreviveu a dois cortes de release porque nada o olhava.

    Um sufixo `.dev` num pacote cuja `main` esta tagueada e uma afirmacao falsa sobre o que aquele
    codigo e — e ela viaja para dentro de cada bundle.
    """
    v = _versao_do_pyproject()
    assert ".dev" not in v, (
        f"versao declarada e {v!r}, um placeholder de desenvolvimento. A `main` deste repositorio "
        "esta tagueada, e cada bundle publicado carrega este valor."
    )
