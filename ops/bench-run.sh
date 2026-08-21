#!/usr/bin/env bash
# Executor de medicao. Roda no HOST DE BENCH, nao na maquina de desenvolvimento.
#
# Regra que organiza o arquivo inteiro, e que custou uma sessao inteira para ser escrita:
#   o que MEDE aborta em erro; o que apenas REGISTRA nunca aborta.
# Toda morte prematura desta sessao foi um comando de registro derrubando a corrida.
set -uo pipefail

SUITE="${SUITE:-analytical/crossover/row-count}"
PROFILE="${PROFILE:-research}"
SMOKE="${SMOKE:-analytical/synthetic/paths}"
TAGS="${TAGS:-base fix}"
PARQUET_DIR=/var/lib/postgresql/theodb-bench-parquet
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

exec > >(tee -a /root/bench-run.log) 2>&1
echo "=== bench-run inicio $(date -Is) suite=$SUITE tags='$TAGS' stamp=$STAMP ==="

# ---------------------------------------------------------------- portao de capacidades
# TODAS as capacidades de uma vez, ANTES de qualquer trabalho caro. Cada linha existe porque a
# ausencia dela ja custou uma corrida:
#   buildx   -> `COPY <<EOF` falha no passo 26/28, DEPOIS de compilar a extensao (~40 min perdidos)
#   psycopg  -> adapter theodb recusa no bootstrap (3 s, mas depois de 18 min de build)
#   schemas  -> arnes nao le o schema de ambiente e invalida a corrida inteira
portao() {
  local falhas=0
  command -v docker >/dev/null || { echo "PORTAO: docker ausente"; falhas=1; }
  docker buildx version >/dev/null 2>&1 || { echo "PORTAO: buildx ausente"; falhas=1; }
  /root/venv/bin/python -c "import psycopg" 2>/dev/null || { echo "PORTAO: psycopg ausente"; falhas=1; }
  /root/venv/bin/python - <<'PY' || falhas=1
import pathlib, sys, theodb_bench
raiz = pathlib.Path(theodb_bench.__file__).resolve().parent.parent
alvo = raiz / "schemas" / "environment.schema.json"
if not alvo.exists():
    print(f"PORTAO: schema de ambiente ausente em {alvo}"); sys.exit(1)
PY
  [ "$falhas" -eq 0 ] || { echo "=== PORTAO REPROVOU — nada caro foi executado ==="; exit 1; }
  echo "=== portao ok $(date -Is) ==="
}

subir() {
  local tag="$1"
  docker rm -f theodb >/dev/null 2>&1 || true
  docker run -d --name theodb -e POSTGRES_HOST_AUTH_METHOD=trust \
    -v /var/run/postgresql:/var/run/postgresql --shm-size=8g \
    "theodb:$tag" \
    -c shared_buffers=16GB -c maintenance_work_mem=8GB \
    -c max_parallel_maintenance_workers=8 -c work_mem=256MB -c max_wal_size=8GB >/dev/null \
    || { echo "FALHA: docker run $tag"; return 1; }

  local pronto=""
  for _ in $(seq 1 120); do
    pg_isready -h /var/run/postgresql -U postgres >/dev/null 2>&1 && { pronto=1; break; }
    sleep 2
  done
  [ -n "$pronto" ] || { echo "FALHA: servidor $tag nao subiu"; docker logs theodb 2>&1 | tail -20; return 1; }

  # O diretorio de Parquet e escrito pelo processo SERVIDOR: existe DENTRO do conteiner e pertence
  # ao usuario do banco. Sem ele `write_parquet` falha, e o arnes reporta `sut_alive` FAIL — culpando
  # o servidor por uma falha que foi de uma consulta. O runbook nao cria este diretorio.
  docker exec -u root theodb mkdir -p "$PARQUET_DIR" || { echo "FALHA: mkdir parquet"; return 1; }
  docker exec -u root theodb chown postgres:postgres "$PARQUET_DIR" || { echo "FALHA: chown parquet"; return 1; }

  # Proveniencia LIDA DO SERVIDOR, nunca da tag da imagem (B-069). Registro puro: nada aborta.
  echo "-- proveniencia $tag --"
  PGUSER=postgres psql -h /var/run/postgresql -tAc "select version()" 2>&1 | head -1 || true
  PGUSER=postgres psql -h /var/run/postgresql -tAc \
    "select extname||' '||extversion from pg_extension where extname like 'theodb%'" 2>&1 | head -3 || true
}

medir() {
  local tag="$1" suite="$2" saida="$3"
  echo "=== $tag :: $suite inicio $(date -Is) ==="
  PGUSER=postgres /root/venv/bin/theodb-bench run "$suite" \
    --system theodb --profile "$PROFILE" --output "$saida"
  local rc=$?
  echo "=== $tag :: $suite fim rc=$rc $(date -Is) ==="
  return $rc
}

portao

# SMOKE PRIMEIRO. Exercita heap+colunar+parquet+oraculo num unico N, em poucos minutos. Se o
# pipeline estiver quebrado, descobre-se aqui — e nao depois de carregar 2 milhoes de linhas
# seis vezes, duas vezes.
PRIMEIRA="${TAGS%% *}"
subir "$PRIMEIRA" || exit 1
if ! medir "$PRIMEIRA" "$SMOKE" "/root/res-$STAMP/smoke"; then
  echo "=== SMOKE REPROVOU — o sweep caro NAO foi executado ==="
  exit 1
fi
echo "=== smoke ok $(date -Is) ==="

for tag in $TAGS; do
  subir "$tag" || exit 1
  medir "$tag" "$SUITE" "/root/res-$STAMP/$tag" || echo "AVISO: $tag terminou nao-zero (bundle preservado)"
done

# Registra QUAL corrida acabou de rodar. Sem isto a coleta faz `tar /root/res-*` e varre tambem os
# resultados de corridas anteriores — inclusive os que vieram DENTRO do snapshot, porque ele foi
# tirado de um host que ja tinha medido. Colher resultado velho junto com novo e pior que nao colher:
# parece completo.
echo "$STAMP" > /root/ULTIMA_CORRIDA

echo "=== FIM $(date -Is) resultados em /root/res-$STAMP ==="
touch /root/PRONTO
