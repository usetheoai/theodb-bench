"""TheoDB Bench -- open, reproducible performance benchmarking for TheoDB."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

try:
    #: Derivado da metadata instalada, que o pip preenche a partir do `pyproject.toml` — UMA fonte.
    #:
    #: Era um literal aqui, ao lado de outro literal no `pyproject.toml`. Os dois diziam
    #: `0.1.0.dev0` e sobreviveram a dois cortes de release sem que nada os olhasse, porque o
    #: `release.yml` deriva a versao da TAG e nunca le nenhum dos dois (B-087).
    #:
    #: Nao e cosmetico: `environment.py` grava `f"theodb-bench {__version__}"` na captura de
    #: ambiente, que vai para dentro do bundle. Um bundle existe para ser reproduzivel por
    #: terceiros, e a versao da ferramenta que o produziu e parte do que torna isso possivel.
    __version__ = _metadata_version("theodb-bench")
except PackageNotFoundError as exc:  # pragma: no cover - so acontece sem instalacao
    # Fail-fast e nao um fallback silencioso: um placeholder aqui reintroduziria exatamente o
    # defeito que este modulo acabou de remover, agora invisivel (`rules/error-handling.md` § 2).
    raise RuntimeError(
        "theodb-bench nao esta instalado neste ambiente, entao a versao nao pode ser determinada. "
        "Instale com `pip install -e .` antes de importar o pacote."
    ) from exc

__all__ = ["__version__"]
