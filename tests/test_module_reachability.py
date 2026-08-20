"""B-071 bullet 4 — um módulo do arnês não pode ficar sem chamador em silêncio.

Medido em 2026-08-17: SEIS módulos estavam implementados e desconectados, incluindo o
núcleo estatístico. `analysis/significance.py`, `analysis/pareto.py` e
`analysis/regression.py` tinham zero importadores em `src/` — código escrito, testado por
unidade, e que nenhuma corrida jamais executava. O `regression_gate` dos perfis `pr`,
`nightly` e `release` prometia detecção de regressão que não acontecia ([[B-072]]).

Três foram ligados. Três continuam órfãos, e este teste é o que impede que a lista cresça
sem alguém decidir.

POR QUE UM BASELINE, e por que ele ENCOLHE. Zerar agora significaria ligar três famílias
de workload inteiras — escopo muito maior que o bullet que pede o portão, e o bullet diz o
que quer: *"o padrão não pode reaparecer em silêncio"*. Três órfãos nomeados num teste não
são silêncio. O teste falha nos dois sentidos: quando aparece um órfão NOVO, e quando um
do baseline foi ligado e não saiu daqui — um baseline que só cresce é uma isenção
permanente com outro nome.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

#: Órfãos conhecidos em 2026-08-20. Cada um é uma família de workload completa cujo
#: caminho até `theodb-bench run` não existe. Sair daqui é o objetivo, não a exceção.
ORFAOS_CONHECIDOS = frozenset({"bench.graph", "bench.operations", "bench.retrieval"})


def _modules_under(*packages: str) -> dict[str, Path]:
    encontrados: dict[str, Path] = {}
    for package in packages:
        for path in (SRC / package).glob("*.py"):
            if path.name != "__init__.py":
                encontrados[f"{package}.{path.stem}"] = path
    return encontrados


def _imported_modules(path: Path) -> set[str]:
    """Módulos `theodb_bench.<pkg>.<nome>` que ESTE arquivo importa.

    Lido por AST e não por substring: `"analysis.regression"` dentro de uma docstring que
    explica o problema contaria como conserto do problema, que é precisamente o defeito
    que esta família de itens registra.
    """
    importados: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            partes = node.module.split(".")
            if partes[:1] == ["theodb_bench"] and len(partes) >= 3:
                importados.add(".".join(partes[1:3]))
            elif partes[:1] == ["theodb_bench"] and len(partes) == 2:
                for alias in node.names:
                    importados.add(f"{partes[1]}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                partes = alias.name.split(".")
                if partes[:1] == ["theodb_bench"] and len(partes) >= 3:
                    importados.add(".".join(partes[1:3]))
    return importados


def _orfaos() -> set[str]:
    modulos = _modules_under("analysis", "bench")
    alcancados: set[str] = set()
    for path in SRC.rglob("*.py"):
        alcancados |= _imported_modules(path) - {name for name, p in modulos.items() if p == path}
    return set(modulos) - alcancados


def test_no_new_orphan_module_appears() -> None:
    novos = sorted(_orfaos() - ORFAOS_CONHECIDOS)
    assert not novos, (
        f"módulo(s) de analysis/ ou bench/ sem importador em src/: {novos}. "
        "Código que nenhuma corrida executa é código que ninguém pode confiar que funciona "
        "— foi assim que o núcleo estatístico ficou desligado por meses (B-071)."
    )


def test_the_baseline_shrinks_and_never_lies() -> None:
    """Um do baseline que ganhou importador tem de SAIR do baseline.

    Sem isto o arquivo vira uma lista de isenções que ninguém revisita, e o portão passa a
    afirmar cobertura sobre módulos que já não precisam dela.
    """
    ligados = sorted(ORFAOS_CONHECIDOS - _orfaos())
    assert not ligados, (
        f"{ligados} já têm importador e continuam listados em ORFAOS_CONHECIDOS. "
        "Remova-os: o baseline existe para encolher."
    )
