"""A guarda de re-execução do `ops/bench-droplet.sh`.

**O defeito que ela existe para impedir, observado em 2026-08-24.** O bash lê um script por
DESLOCAMENTO DE BYTE, não de uma vez. Acrescentar uma variável no topo do arquivo com uma corrida em
voo desloca todo o resto, e o interpretador retoma no meio de uma linha:

    ops/bench-droplet.sh: linha 274: erro de sintaxe próximo ao token inesperado `('
    ops/bench-droplet.sh: linha 274: `  echo "=== theodb:$nome pronto (de $sha) $(date -Is) ==="`

As duas pernas de um A/B morreram assim, depois de os dois binários já terem sido construídos, e o
`bash -n` rodado em seguida **passava** — o arquivo estava correto; o processo é que tinha perdido a
posição. Um teste que só checasse sintaxe não pegaria nada.

**Limite honesto sobre o que este arquivo prova.** Tentei um teste que reproduzisse a falha:
executar um script sem a guarda e editá-lo em voo. **Não reproduz** — nem com 32 KiB de enchimento
empurrando o comando final para longe do começo. O bash já tinha lido o suficiente. A falha real
aconteceu numa corrida de dezenas de minutos, com o interpretador precisando de mais entrada depois
de muito tempo parado; sinteticamente essa janela não se recria de forma confiável.

Um teste positivo naquelas condições passaria com ou sem a guarda, o que é teatro. O que ficou é
determinístico e testa o MECANISMO: com a guarda, o script executa a partir de uma cópia, então
editar o original deixa de poder alcançá-lo — seja qual for o tamanho do bloco de leitura do bash.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DROPLET = RAIZ / "ops" / "bench-droplet.sh"


def _guarda() -> str:
    """O bloco de re-execução, lido do arquivo real — para o teste cair se ele for removido."""
    texto = DROPLET.read_text(encoding="utf-8")
    ini = texto.index('if [ -z "${BENCH_DROPLET_COPIA:-}" ]; then')
    fim = texto.index("\nfi\n", ini) + len("\nfi\n")
    return texto[ini:fim]


def test_a_guarda_de_reexecucao_esta_no_script_real():
    assert 'if [ -z "${BENCH_DROPLET_COPIA:-}" ]; then' in DROPLET.read_text(encoding="utf-8")
    assert "mktemp" in _guarda()


def _rodar(tmp_path: Path, com_guarda: bool) -> tuple[str, Path]:
    alvo = tmp_path / ("com.sh" if com_guarda else "sem.sh")
    corpo = textwrap.dedent(
        """
        echo "EXECUTANDO_DE=$0"
        echo "COPIAS=$(ls "$TMPDIR_TESTE"/bench-droplet.* 2>/dev/null | wc -l)"
        """
    )
    alvo.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + (_guarda() if com_guarda else "") + corpo,
        encoding="utf-8",
    )
    os.chmod(alvo, 0o755)
    env = {**os.environ, "TMPDIR": str(tmp_path), "TMPDIR_TESTE": str(tmp_path)}
    saida = subprocess.run(
        ["bash", str(alvo)], capture_output=True, text=True, env=env, timeout=30
    )
    assert saida.returncode == 0, saida.stdout + saida.stderr
    return saida.stdout, alvo


def test_com_a_guarda_o_script_executa_de_uma_COPIA(tmp_path: Path):
    """É isto que torna a edição em voo inofensiva: o arquivo que o bash está lendo não é mais o
    que alguém edita."""
    saida, alvo = _rodar(tmp_path, com_guarda=True)
    executado = next(l.split("=", 1)[1] for l in saida.splitlines() if l.startswith("EXECUTANDO_DE="))
    assert executado != str(alvo), "executou do proprio arquivo — a guarda nao teve efeito"
    assert "bench-droplet." in executado
    assert next(l for l in saida.splitlines() if l.startswith("COPIAS=")) == "COPIAS=1"


def test_sem_a_guarda_o_script_executa_do_PROPRIO_arquivo(tmp_path: Path):
    """O controle. Sem ele, o teste acima não distinguiria a guarda funcionando de o bash já se
    comportar assim sozinho."""
    saida, alvo = _rodar(tmp_path, com_guarda=False)
    executado = next(l.split("=", 1)[1] for l in saida.splitlines() if l.startswith("EXECUTANDO_DE="))
    assert executado == str(alvo)


def test_a_copia_e_removida_ao_fim(tmp_path: Path):
    """Uma corrida por hora deixando lixo em /tmp seria um vazamento lento e silencioso."""
    _rodar(tmp_path, com_guarda=True)
    assert list(tmp_path.glob("bench-droplet.*")) == []


def test_nenhum_caminho_e_derivado_de_dollar_zero(tmp_path: Path):
    """A guarda faz `$0` apontar para /tmp, então derivar caminho dele passou a estar ERRADO.

    Medido em 2026-08-24: a primeira versão da guarda quebrou o portão de refs — ele reportou
    `nao resolve em ` com caminho VAZIO, porque `$(dirname "$0")/../../theo-db` a partir de /tmp não
    existe. O portão estava certo; o caminho é que tinha sumido. Cinco lugares faziam isso.
    """
    texto = DROPLET.read_text(encoding="utf-8")
    codigo = [
        linha
        for linha in texto.splitlines()
        if not linha.lstrip().startswith("#") and 'dirname "$0"' in linha
    ]
    assert codigo == [], f"ainda derivam caminho de $0: {codigo}"


def test_AQUI_aponta_para_o_diretorio_real_mesmo_executando_da_copia(tmp_path: Path):
    """O que substituiu `dirname "$0"` tem de sobreviver à re-execução — senão o conserto só troca
    um caminho errado por outro."""
    real = tmp_path / "ops"
    real.mkdir()
    alvo = real / "corrida.sh"
    alvo.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        'AQUI="${BENCH_DROPLET_AQUI:-$(cd "$(dirname "$0")" && pwd)}"\n'
        'export BENCH_DROPLET_AQUI="$AQUI"\n'
        + _guarda()
        + 'echo "AQUI=$AQUI"\necho "ZERO=$0"\n',
        encoding="utf-8",
    )
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    saida = subprocess.run(
        ["bash", str(alvo)], capture_output=True, text=True, env=env, timeout=30
    )
    assert saida.returncode == 0, saida.stdout + saida.stderr
    linhas = dict(l.split("=", 1) for l in saida.stdout.splitlines() if "=" in l)
    assert linhas["AQUI"] == str(real), saida.stdout
    assert linhas["ZERO"] != str(alvo), "executou do proprio arquivo — a guarda nao teve efeito"
