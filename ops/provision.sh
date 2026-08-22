#!/usr/bin/env bash
# Provisiona um HOST DE BENCH do zero. Este arquivo e a FONTE DE VERDADE do que a maquina de
# medicao precisa ter; o snapshot DigitalOcean e apenas um cache derivado dele.
#
# A inversao e deliberada. Um snapshot e um blob opaco: em tres meses ninguem sabe o que tem
# dentro, e o projeto inteiro se apoia no principio oposto — ate `shared_buffers` e declarado,
# nao default, porque artefato com estado nao-declarado nao e reproduzivel. Se o snapshot sumir
# ou envelhecer, roda-se este script e ele volta.
#
#   ./provision.sh            provisiona
#   ./provision.sh --verify   so verifica capacidades (idempotente, barato, seguro em prod)
#
# Cada capacidade verificada aqui custou uma corrida perdida em 2026-08-21. Ver
# `wiki/runbooks/droplet-de-medicao.md` no theo-db para o relato de cada uma.
set -uo pipefail

BENCH_SRC="${BENCH_SRC:-/root/bench}"
VENV="${VENV:-/root/venv}"

verificar() {
  local falhas=0
  echo "--- capacidades ---"
  _c() { # nome, comando
    if eval "$2" >/dev/null 2>&1; then printf '  OK    %s\n' "$1"
    else printf '  FALTA %s\n' "$1"; return 1; fi
  }
  _c "docker"          "command -v docker"                        || falhas=1
  _c "buildx"          "docker buildx version"                    || falhas=1
  _c "psql"            "command -v psql"                          || falhas=1
  _c "git"             "command -v git"                           || falhas=1
  _c "venv"            "test -x $VENV/bin/python"                 || falhas=1
  _c "theodb-bench"    "test -x $VENV/bin/theodb-bench"           || falhas=1
  _c "psycopg"         "$VENV/bin/python -c 'import psycopg'"     || falhas=1
  _c "socket pg"       "test -d /var/run/postgresql"              || falhas=1

  # O arnes le `schemas/` AO LADO do pacote. Um `pip install` de tarball nao leva esse diretorio,
  # e a corrida so descobre no fim, invalidando o bundle inteiro. Instalacao editavel resolve.
  if $VENV/bin/python - <<'PY' >/dev/null 2>&1
import pathlib, sys, theodb_bench
raiz = pathlib.Path(theodb_bench.__file__).resolve().parent.parent
sys.exit(0 if (raiz / "schemas" / "environment.schema.json").exists() else 1)
PY
  then printf '  OK    schemas do arnes\n'
  else printf '  FALTA schemas do arnes (instale com pip install -e)\n'; falhas=1; fi

  echo "--- higiene de medicao (advisory) ---"
  local gov; gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo indisponivel)
  printf '  governor: %s\n' "$gov"
  printf '  swap: %s\n' "$(swapon --show --noheadings 2>/dev/null | wc -l) dispositivo(s)"

  return $falhas
}

if [ "${1:-}" = "--verify" ]; then verificar; exit $?; fi

echo "=== provisionando $(date -Is) ==="
export DEBIAN_FRONTEND=noninteractive

# ESPERAR O BOOT TERMINAR antes de tocar no apt.
#
# MEDIDO em 2026-08-21, e so a execucao de verdade revelou: numa VM recem-criada o `cloud-init` e o
# `unattended-upgrades` seguram o lock do dpkg nos primeiros minutos. O `apt-get install` falha, e o
# provisionamento inteiro passou em 7 SEGUNDOS reportando sucesso parcial — docker ausente, e a causa
# invisivel. Num droplet menor o mesmo script funcionou, porque o boot ja tinha terminado quando ele
# chegou aqui: uma corrida de largada disfarcada de bug intermitente.
command -v cloud-init >/dev/null 2>&1 && cloud-init status --wait >/dev/null 2>&1
for _ in $(seq 1 60); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  echo "    aguardando o lock do dpkg..."
  sleep 5
done

# E FALHAR ALTO se o apt falhar. A versao anterior seguia adiante e deixava o portao reportar a
# consequencia ("FALTA docker") em vez da causa ("apt nao rodou") — mandando quem diagnostica para o
# lado errado, que e o mesmo defeito do `sut_alive` do arnes.
apt-get update -qq || { echo "FALHA: apt-get update nao completou"; exit 1; }
# docker-buildx e SEPARADO de docker.io no Ubuntu 24.04. Sem ele o `docker build` cai no builder
# legado, que nao entende `COPY <<EOF` (heredoc) — e falha no PASSO 26 DE 28, depois de compilar.
apt-get install -y -qq docker.io docker-buildx python3-venv python3-pip postgresql-client git >/dev/null \
  || { echo "FALHA: apt-get install nao completou"; exit 1; }

# Higiene de medicao: paginacao e escalonamento de frequencia distorcem latencia de cauda.
swapoff -a 2>/dev/null || true
cpupower frequency-set --governor performance >/dev/null 2>&1 || echo "governor: nao ajustavel neste host"

# O arnes NAO sobe servidor — ele mede um que ja exista, pelo socket unix.
#
# MEDIDO em 2026-08-21: `/var/run` e TMPFS. Nada ali sobrevive a um boot, e portanto nada ali entra
# num snapshot — foi a unica das nove capacidades que um host nascido do snapshot NAO tinha. Nao e
# defeito do snapshot; e o Linux funcionando como projetado, e supor persistencia ali era erro meu.
# `tmpfiles.d` e o mecanismo NATIVO do systemd para exatamente isto (degrau 3 da parsimony ladder),
# entao o diretorio volta a cada boot sem reprovisionar.
mkdir -p /var/run/postgresql && chmod 777 /var/run/postgresql
mkdir -p /usr/lib/tmpfiles.d
printf 'd /var/run/postgresql 0777 root root -\n' > /usr/lib/tmpfiles.d/theodb-bench.conf

if [ -d "$BENCH_SRC" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  # EDITAVEL, de proposito: mantem o pacote enraizado em $BENCH_SRC, onde `schemas/` existe.
  "$VENV/bin/pip" install -q -e "$BENCH_SRC[postgres,datasets]" \
    || { echo "FALHA: pip install do arnes"; exit 1; }
else
  echo "AVISO: $BENCH_SRC ausente — venv nao criado. Envie o arnes e rode de novo."
fi

echo "=== verificando $(date -Is) ==="
verificar
