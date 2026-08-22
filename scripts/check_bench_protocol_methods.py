#!/usr/bin/env python3
"""Método público de benchmark que não está no protocolo e ninguém chama.

POR QUE ESTE CHECADOR EXISTE, E POR QUE O `vulture` NÃO BASTA. Medido em 2026-08-22:
`RetrievalBenchmark.summary()` monta a comparação entre caminhos e tem **zero chamadores**
— `runner.py` pede ao benchmark apenas `load` e `points`, e `bench/protocol.py` declara só
esses dois. O payload que ela produz nunca entrou em bundle nenhum.

O `vulture` **não a encontra**, e a razão é estrutural: ele casa por NOME. O nome `summary`
aparece 27 vezes em `src/` — outro método `summary()` em `analysis/significance.py`, uma
variável local em `analysis/statistics.py` —, e qualquer uma dessas ocorrências marca todas
as definições do nome como usadas. O mesmo cega `assert_analytical_path`, cujo único "uso"
era o `super()` dentro do próprio override.

Este checador é estreito de propósito: ele olha uma família de classes e uma pergunta
— *este método está no protocolo, ou alguém o chama?* Não substitui o `vulture`; cobre o
ponto cego dele.

LIMITES DECLARADOS, e o segundo é o mesmo defeito que este arquivo critica:

1. A busca por chamador é textual (`.nome(`). Um despacho dinâmico (`getattr(obj, nome)()`)
   passa despercebido.
2. **Ela casa por NOME, como o `vulture`.** Uma chamada `.summary(` a um `summary()` de
   OUTRA classe conta como chamador — e é exatamente o caso que motivou este arquivo:
   `analysis/significance.py` tem o seu, e `tests/test_analysis_significance.py` o chama.
   Portanto este checador **não teria encontrado** `RetrievalBenchmark.summary()`; o que ele
   cobre é o caso em que o nome é único, que é a maioria.
3. Um método chamado **só por teste** conta como chamado. A pergunta que ele responde é
   "alguém o exercita?", não "ele chega a um artefato?" — e as duas são diferentes:
   `RetrievalBenchmark.summary()` tem teste e nenhum chamador de produção, então o payload
   que ela monta nunca entrou em bundle algum.

Por isso o veredito é AVISO e não bloqueio: um portão que reprova sobre heurística de nome
gasta mais confiança do que compra.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

#: Sufixos das classes que implementam a família de benchmark.
FAMILIA = ("Benchmark", "Workload")


def metodos_publicos(caminho: Path) -> dict[str, list[tuple[str, int]]]:
    arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    achados: dict[str, list[tuple[str, int]]] = {}
    for no in ast.walk(arvore):
        if not isinstance(no, ast.ClassDef) or not no.name.endswith(FAMILIA):
            continue
        for corpo in no.body:
            if isinstance(corpo, ast.FunctionDef) and not corpo.name.startswith("_"):
                achados.setdefault(no.name, []).append((corpo.name, corpo.lineno))
    return achados


def declarados_no_protocolo(protocolo: Path) -> set[str]:
    if not protocolo.exists():
        return set()
    arvore = ast.parse(protocolo.read_text(encoding="utf-8"))
    return {
        no.name
        for no in ast.walk(arvore)
        if isinstance(no, ast.FunctionDef) and not no.name.startswith("_")
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz", type=Path, default=Path.cwd())
    args = p.parse_args()

    bench = args.raiz / "src" / "bench"
    if not bench.is_dir():
        print(f"sem {bench}; nada a checar")
        return 0

    protocolo = declarados_no_protocolo(bench / "protocol.py")
    corpo = "\n".join(
        f.read_text(encoding="utf-8")
        for f in list((args.raiz / "src").rglob("*.py")) + list((args.raiz / "tests").rglob("*.py"))
        if f.is_file()
    )

    orfaos: list[str] = []
    for arquivo in sorted(bench.glob("*.py")):
        if arquivo.name == "protocol.py":
            continue
        for classe, metodos in metodos_publicos(arquivo).items():
            for nome, linha in metodos:
                if nome in protocolo:
                    continue
                # Chamada por atributo, em qualquer lugar de src/ ou tests/.
                if re.search(rf"\.{re.escape(nome)}\s*\(", corpo):
                    continue
                orfaos.append(f"{arquivo.relative_to(args.raiz)}:{linha}  {classe}.{nome}()")

    if not orfaos:
        print("OK: todo metodo publico de benchmark esta no protocolo ou tem chamador")
        return 0
    print("AVISO: metodo publico de benchmark fora do protocolo e sem chamador encontrado:")
    for o in orfaos:
        print(f"  {o}")
    print(
        "\nUm metodo assim nao aparece em bundle nenhum. Ou ele entra no protocolo, ou sai —\n"
        "manter um que parece cobrir uma lacuna e a unica resposta que nao e honesta.\n"
        "Busca por chamador e textual: despacho dinamico passa, e por isso isto e AVISO."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
