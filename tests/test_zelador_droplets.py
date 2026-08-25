"""O zelador destrói droplet órfão e NÃO toca em corrida viva.

**O defeito que ele existe para impedir, medido em 2026-08-25.** O `trap EXIT` do `bench-droplet.sh`
só dispara quando o script SAI — e ele não sai quando a árvore de processos é morta de uma vez (fim de
sessão, SIGKILL, terminal fechado). Um droplet ficou **32 minutos** de pé, US$ 0,41, porque a sessão
que o criou terminou antes da corrida.

A guarda é um marcador `PID ID` por droplet, varrido no início de cada corrida. **Sem heurística de
tempo**: uma corrida legítima de três horas noutro terminal tem PID vivo e não é tocada — e é isso que
estes testes provam, um caso de cada.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DROPLET = RAIZ / "ops" / "bench-droplet.sh"


def _zelador() -> str:
    """O bloco do zelador, lido do arquivo real — cai se alguém o remover."""
    texto = DROPLET.read_text(encoding="utf-8")
    ini = texto.index('MARCADORES="${MARCADORES:-')
    fim = texto.index("\ndone\n", ini) + len("\ndone\n")
    return texto[ini:fim]


def _rodar(tmp_path: Path, pid: str) -> tuple[str, Path]:
    """Monta um marcador com o PID dado e roda o zelador com um `doctl` falso."""
    marcadores = tmp_path / "marcas"
    marcadores.mkdir()
    (marcadores / "999.marca").write_text(f"{pid} 999 theo-bench-teste\n", encoding="utf-8")

    falso = tmp_path / "doctl"
    falso.write_text("#!/usr/bin/env bash\necho \"DESTRUIU $*\" >> \"$TESTE_LOG\"\n", encoding="utf-8")
    os.chmod(falso, 0o755)

    script = tmp_path / "z.sh"
    script.write_text("#!/usr/bin/env bash\nset -uo pipefail\n" + _zelador(), encoding="utf-8")
    log = tmp_path / "chamadas.log"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "MARCADORES": str(marcadores),
        "TESTE_LOG": str(log),
        "HOME": str(tmp_path),
    }
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    return (log.read_text(encoding="utf-8") if log.exists() else ""), marcadores


def test_pid_morto_tem_o_droplet_destruido(tmp_path: Path) -> None:
    # PID improvável de existir; confirmado morto antes de usar.
    morto = "999999"
    assert not Path(f"/proc/{morto}").exists(), "escolha um PID que nao exista"
    chamadas, marcadores = _rodar(tmp_path, morto)
    assert "DESTRUIU" in chamadas and "999" in chamadas, f"nao destruiu o orfao: {chamadas!r}"
    assert list(marcadores.glob("*.marca")) == [], "o marcador do orfao deveria sumir"


def test_pid_vivo_NAO_e_tocado(tmp_path: Path) -> None:
    """O controle. Sem ele, um zelador que destruísse TUDO passaria no teste acima."""
    chamadas, marcadores = _rodar(tmp_path, str(os.getpid()))
    assert chamadas == "", f"tocou numa corrida viva: {chamadas!r}"
    assert len(list(marcadores.glob("*.marca"))) == 1, "o marcador da corrida viva deveria ficar"


def test_o_bloco_existe_no_script_real() -> None:
    texto = DROPLET.read_text(encoding="utf-8")
    assert "ZELADOR" in texto and 'echo "$$ $ID $NOME" > "$MARCA"' in texto
