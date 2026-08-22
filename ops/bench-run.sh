#!/usr/bin/env bash
# Executor de medicao. Roda no HOST DE BENCH, nao na maquina de desenvolvimento.
#
# Regra que organiza o arquivo inteiro, e que custou uma sessao inteira para ser escrita:
#   o que MEDE aborta em erro; o que apenas REGISTRA nunca aborta.
# Toda morte prematura desta sessao foi um comando de registro derrubando a corrida.
set -uo pipefail

SUITE="${SUITE:-analytical/crossover/row-count}"
PROFILE="${PROFILE:-research}"
# Isolamento DECLARADO. Os perfis `nightly` e `release` exigem `cpu_limit` e `memory_limit`, e sem
# declaracao eles saem UNAVAILABLE e invalidam a corrida — em qualquer hardware. Vazio = nao declara,
# que e legitimo em `research` e honesto: inventar um default esconderia que nada foi declarado.
CPU_SET="${CPU_SET:-}"
MEM_MAX="${MEM_MAX:-}"
# MODE=contention roda o executor de contencao escrita x scan em vez de uma suite registrada.
# O `theodb-bench contention` ASSUME a tabela pronta e trata `--regime` como DECLARACAO: quem roda
# tem de torna-la verdadeira. Por isso os dois regimes usam a MESMA carga e servidores com
# `shared_buffers` diferentes — residencia em cache e o que separa os dois, nao o tamanho absoluto.
MODE="${MODE:-suite}"
# MODE=headtohead sobe TheoDB, AlloyDB Omni e pgvector no MESMO host e mede os tres na mesma suite.
# O Omni ganha memoria explicita porque ele tem um matador de backends proprio (`g_term_it.cc`) que
# encerra consultas quando JULGA a memoria critica — medido em 2026-08-22, ele matou uma consulta
# vetorial com o backend em 16 MB numa maquina apertada. Dar folga nao e favorecer: e medir o produto
# em vez de medir o guarda dele.
OMNI_IMAGE="${OMNI_IMAGE:-google/alloydbomni:latest}"
PGVECTOR_IMAGE="${PGVECTOR_IMAGE:-pgvector/pgvector:pg17}"
# MEDIDO em 2026-08-22: 1M linhas de `(id, value)` no colunar ocupam **3.248 kB** — a compressao e
# tao boa que o regime `exceeds-cache` com 32 MB de `shared_buffers` nao excedia NADA. Declarar um
# regime nao o torna verdadeiro, e medir "fora do cache" com o dado inteiro dentro dele mediria a
# mesma coisa duas vezes com rotulos diferentes. 40M linhas dao ~130 MB, que excede os 32 MB com
# folga; o regime residente usa a mesma carga com 16 GB, onde ela cabe inteira.
CONT_LINHAS="${CONT_LINHAS:-40000000}"
CONT_LEITORES="${CONT_LEITORES:-4}"
CONT_ESCRITORES="${CONT_ESCRITORES:-2}"
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
  # Sob MEM_MAX o arnes roda DENTRO de um cgroup com limite, porque e o que ele exige para marcar
  # `memory_limit` como respeitado — aplicar o limite ele mesmo pediria privilegio e teria efeito
  # colateral sobre o host, entao ele LE o limite que ja vale. `systemd-run --scope` e o mecanismo
  # nativo para criar esse cgroup (degrau 3 da parsimony ladder), e sem ele os perfis `nightly` e
  # `release` sao inalcancaveis.
  if [ -n "$MEM_MAX" ] && command -v systemd-run >/dev/null 2>&1; then
    PGUSER=postgres systemd-run --scope --quiet -p "MemoryMax=$MEM_MAX" \
      /root/venv/bin/theodb-bench run "$suite" \
      --system theodb --profile "$PROFILE" --output "$saida" \
      ${CPU_SET:+--cpu-set "$CPU_SET"} --memory "$MEM_MAX"
  else
    PGUSER=postgres /root/venv/bin/theodb-bench run "$suite" \
      --system theodb --profile "$PROFILE" --output "$saida" \
      ${CPU_SET:+--cpu-set "$CPU_SET"} ${MEM_MAX:+--memory "$MEM_MAX"}
  fi
  local rc=$?
  echo "=== $tag :: $suite fim rc=$rc $(date -Is) ==="
  return $rc
}

# Sobe um concorrente em container proprio, um porto TCP cada. Vive aqui, e nao dentro
# de um modo, porque `headtohead` e `tpch` precisam da mesma coisa (DRY).
subir_externo() {
  local nome="$1" imagem="$2" porta="$3"
  docker rm -f "$nome" >/dev/null 2>&1 || true
  docker run -d --name "$nome" -e POSTGRES_PASSWORD=x -e POSTGRES_HOST_AUTH_METHOD=trust \
    -p "$porta":5432 --shm-size=4g "$imagem" >/dev/null || { echo "FALHA: docker run $nome"; return 1; }
  # Prontidao lida de FORA, por TCP — nao por `docker exec pg_isready`.
  #
  # Medido no Omni em 2026-08-22, amostrando os dois predicados a cada 3 s:
  #
  #   18s  docker exec: SIM   tcp: nao   <- servidor TEMPORARIO do entrypoint,
  #                                         que so escuta no socket unix
  #   21s  docker exec: nao   tcp: nao   <- o init derruba o temporario
  #   27s  docker exec: SIM   tcp: SIM   <- servidor real
  #
  # Um script que avanca aos 18 s conversa com um servidor que vai ser DESCARTADO:
  # o `ALTER SYSTEM` some junto e ninguem reclama. Foi assim que o primeiro teste
  # desta funcao falhou. O porto TCP so e publicado pelo servidor real, entao ele
  # e o unico dos dois que responde a pergunta que se quis fazer.
  for _ in $(seq 1 120); do
    PGPASSWORD=x pg_isready -h 127.0.0.1 -p "$porta" -U postgres >/dev/null 2>&1 && return 0
    sleep 3
  done
  echo "FALHA: $nome nao subiu"; docker logs "$nome" 2>&1 | tail -15; return 1
}

portao

# ---------------------------------------------------------------- tres vias (B-059 bullet 4)
if [ "$MODE" = "headtohead" ]; then

  # O engine colunar do Omni vem DESLIGADO e o GUC e de contexto `postmaster`: nao ha
  # SET de sessao que o ligue. Sem isto toda consulta colunar cai para heap, e o portao
  # do adapter aborta a corrida com essa mensagem exata — que e o portao funcionando,
  # mas custa um droplet inteiro para descobrir. Ligar aqui e a metade facil; a metade
  # que importa e VERIFICAR no servidor, porque `ALTER SYSTEM` sem restart nao aplica e
  # nao reclama (B-058).
  ligar_colunar_omni() {
    docker exec -u postgres omni psql -q -c \
      "ALTER SYSTEM SET google_columnar_engine.enabled = on" >/dev/null 2>&1 \
      || { echo "FALHA: ALTER SYSTEM do google_columnar_engine"; return 1; }
    docker restart omni >/dev/null 2>&1 || { echo "FALHA: restart do omni"; return 1; }
    local pronto=nao
    for _ in $(seq 1 120); do
      # TCP, pelo mesmo motivo medido em `subir_externo`.
      PGPASSWORD=x pg_isready -h 127.0.0.1 -p 55460 -U postgres >/dev/null 2>&1 \
        && { pronto=sim; break; }
      sleep 3
    done
    [ "$pronto" = sim ] || { echo "FALHA: omni nao voltou apos restart"; return 1; }
    local v
    v=$(docker exec -u postgres omni psql -tAc \
      "SHOW google_columnar_engine.enabled" 2>/dev/null | tr -d '[:space:]')
    [ "$v" = "on" ] || {
      echo "FALHA: google_columnar_engine.enabled = '${v:-<vazio>}' depois de ALTER SYSTEM + restart"
      return 1
    }
    echo "-- omni: google_columnar_engine.enabled = on (lido do servidor) --"
  }

  docker pull "$OMNI_IMAGE" >/dev/null 2>&1 || { echo "FALHA: pull do Omni"; exit 1; }
  docker pull "$PGVECTOR_IMAGE" >/dev/null 2>&1 || { echo "FALHA: pull do pgvector"; exit 1; }

  # TheoDB pelo socket unix (como as outras suites); os outros dois por TCP, um porto cada.
  subir "${TAGS%% *}" || exit 1
  subir_externo omni "$OMNI_IMAGE" 55460 || exit 1
  # So para as suites analiticas: ligar o engine custa memoria e um restart, e uma
  # corrida vetorial nao o usa.
  case "$SUITE" in
    analytical/*) ligar_colunar_omni || exit 1 ;;
  esac
  subir_externo pgv "$PGVECTOR_IMAGE" 55461 || exit 1

  echo "-- proveniencia dos TRES, lida de cada servidor --"
  PGUSER=postgres psql -h /var/run/postgresql -tAc "select 'theodb: '||version()" 2>&1 | head -1 || true
  PGPASSWORD=x psql -h 127.0.0.1 -p 55460 -U postgres -tAc "select 'omni:   '||version()" 2>&1 | head -1 || true
  PGPASSWORD=x psql -h 127.0.0.1 -p 55461 -U postgres -tAc "select 'pgv:    '||version()" 2>&1 | head -1 || true

  for alvo in "theodb::" "alloydbomni:127.0.0.1:55460" "pgvector:127.0.0.1:55461"; do
    sist="${alvo%%:*}"; resto="${alvo#*:}"; host="${resto%%:*}"; porta="${resto#*:}"
    echo "=== $sist :: $SUITE inicio $(date -Is) ==="
    if [ -n "$host" ]; then
      PGHOST="$host" PGPORT="$porta" PGUSER=postgres PGPASSWORD=x \
        /root/venv/bin/theodb-bench run "$SUITE" --system "$sist" --profile "$PROFILE" \
        --output "/root/res-$STAMP/$sist"
    else
      PGUSER=postgres /root/venv/bin/theodb-bench run "$SUITE" --system "$sist" \
        --profile "$PROFILE" --output "/root/res-$STAMP/$sist"
    fi
    echo "=== $sist fim rc=$? $(date -Is) ==="
  done
  echo "$STAMP" > /root/ULTIMA_CORRIDA
  echo "=== FIM $(date -Is) resultados em /root/res-$STAMP ==="
  touch /root/PRONTO
  exit 0
fi

# ---------------------------------------------------------------- TPC-H (B-058 bullet 1)
#
# O criterio: "TPC-H nos mesmos moldes: theodb_columnar contra heap no MESMO binario, e
# contra o Omni com engine off/on na mesma maquina". Quatro corridas, um host, um dado.
#
# O engine do Omni e ligado DEPOIS das corridas com ele desligado, e nao antes, porque
# `enabled` e de contexto postmaster: liga-lo exige restart, e um restart no meio de uma
# corrida invalidaria a que estivesse rodando.
if [ "$MODE" = "tpch" ]; then
  SFS="${SFS:-0.01 0.1}"
  DEST="/root/res-$STAMP/tpch"
  mkdir -p "$DEST"

  rodar_tpch() {
    local rotulo="$1" sistema="$2" caminho="$3" dsn="$4" sf="$5"
    local saida="$DEST/${rotulo}-sf${sf}.json"
    echo "=== tpch $rotulo sf=$sf inicio $(date -Is) ==="
    if [ -n "$dsn" ]; then
      /root/venv/bin/theodb-bench tpch --system "$sistema" --dsn "$dsn" \
        --scale-factor "$sf" --path "$caminho" > "$saida" 2>"$saida.err"
    else
      PGUSER=postgres /root/venv/bin/theodb-bench tpch --system "$sistema" \
        --scale-factor "$sf" --path "$caminho" > "$saida" 2>"$saida.err"
    fi
    local rc=$?
    # O que MEDE aborta em erro; o que apenas REGISTRA nunca aborta. Uma perna que caiu
    # e um dado sobre o sistema, e some se a corrida inteira morrer com ela.
    if [ "$rc" -ne 0 ]; then
      echo "AVISO: $rotulo sf=$sf rc=$rc"; head -3 "$saida.err" 2>/dev/null
    else
      python3 - "$saida" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
for q, v in sorted(d["queries"].items()):
    marca = "" if v["matches_oracle"] else "  <-- DISCORDA DO ORACULO"
    print(f"  {q:4} {v['seconds']*1000:9.1f} ms  {v['rows_returned']:6d} linhas{marca}")
PYEOF
    fi
    echo "=== tpch $rotulo sf=$sf fim rc=$rc $(date -Is) ==="
  }

  docker pull "$OMNI_IMAGE" >/dev/null 2>&1 || { echo "FALHA: pull do Omni"; exit 1; }
  subir "${TAGS%% *}" || exit 1
  subir_externo omni "$OMNI_IMAGE" 55460 || exit 1

  echo "-- proveniencia, lida de cada servidor --"
  PGUSER=postgres psql -h /var/run/postgresql -tAc "select 'theodb: '||version()" 2>&1 | head -1 || true
  PGPASSWORD=x psql -h 127.0.0.1 -p 55460 -U postgres -tAc "select 'omni:   '||version()" 2>&1 | head -1 || true
  OMNI_DSN="postgresql://postgres:x@127.0.0.1:55460/postgres"

  for sf in $SFS; do
    rodar_tpch "theodb-heap"     theodb      row      ""          "$sf"
    rodar_tpch "theodb-colunar"  theodb      columnar ""          "$sf"
    rodar_tpch "omni-engineoff"  alloydbomni row      "$OMNI_DSN" "$sf"
  done

  # Agora sim: liga o engine (ALTER SYSTEM + restart) e repete o lado do Omni.
  docker exec -u postgres omni psql -q -c \
    "ALTER SYSTEM SET google_columnar_engine.enabled = on" >/dev/null 2>&1 \
    || { echo "FALHA: ALTER SYSTEM"; exit 1; }
  docker restart omni >/dev/null 2>&1 || { echo "FALHA: restart do omni"; exit 1; }
  for _ in $(seq 1 120); do
    PGPASSWORD=x pg_isready -h 127.0.0.1 -p 55460 -U postgres >/dev/null 2>&1 && break
    sleep 3
  done
  V=$(docker exec -u postgres omni psql -tAc "SHOW google_columnar_engine.enabled" 2>/dev/null | tr -d '[:space:]')
  [ "$V" = "on" ] || { echo "FALHA: engine = '${V:-<vazio>}' apos ALTER SYSTEM + restart"; exit 1; }
  echo "-- omni: google_columnar_engine.enabled = on (lido do servidor) --"

  for sf in $SFS; do
    rodar_tpch "omni-engineon-heap"    alloydbomni row      "$OMNI_DSN" "$sf"
    rodar_tpch "omni-engineon-colunar" alloydbomni columnar "$OMNI_DSN" "$sf"
  done

  echo "$STAMP" > /root/ULTIMA_CORRIDA
  echo "=== FIM $(date -Is) resultados em /root/res-$STAMP ==="
  touch /root/PRONTO
  exit 0
fi

# ---------------------------------------------------------------- contencao (B-058 bullet 3)
if [ "$MODE" = "contention" ]; then
  for regime in memory-resident exceeds-cache; do
    # `exceeds-cache` nao vem de mais dados, vem de MENOS cache: mesma tabela, `shared_buffers`
    # pequeno. Declarar o regime e faze-lo valer sao coisas diferentes, e o arnes so registra a
    # declaracao — torna-la verdadeira e responsabilidade de quem mede.
    case "$regime" in
      memory-resident) SB=16GB ;;
      exceeds-cache)   SB=32MB ;;
    esac
    echo "=== contencao :: $regime (shared_buffers=$SB) inicio $(date -Is) ==="
    docker rm -f theodb >/dev/null 2>&1 || true
    docker run -d --name theodb -e POSTGRES_HOST_AUTH_METHOD=trust \
      -v /var/run/postgresql:/var/run/postgresql --shm-size=8g \
      "theodb:${TAGS%% *}" -c shared_buffers=$SB -c work_mem=64MB >/dev/null || { echo "FALHA: docker run"; exit 1; }
    pronto=""
    for _ in $(seq 1 120); do
      pg_isready -h /var/run/postgresql -U postgres >/dev/null 2>&1 && { pronto=1; break; }
      sleep 2
    done
    [ -n "$pronto" ] || { echo "FALHA: servidor nao subiu"; exit 1; }

    # `-v ON_ERROR_STOP=1` NAO e detalhe. Sem ele o `psql` devolve 0 mesmo quando o SQL falha, e a
    # carga "tem sucesso" com a tabela inexistente — foi o que aconteceu em 2026-08-22: os dois
    # regimes rodaram, leram 0/200, e o erro real nunca apareceu. E a MESMA armadilha que derrubou
    # uma corrida mais cedo hoje, com o nome da extensao, reintroduzida em codigo novo.
    if ! PGUSER=postgres psql -h /var/run/postgresql -v ON_ERROR_STOP=1 -q -c \
      "CREATE TABLE bench_contention (id bigint, value bigint) USING theodb_columnar;
       INSERT INTO bench_contention SELECT g, g FROM generate_series(1,$CONT_LINHAS) g;"; then
      echo "FALHA: carga de $CONT_LINHAS linhas nao completou (erro acima)"; exit 1
    fi
    echo "-- $regime: $CONT_LINHAS linhas, shared_buffers=$SB --"
    # O tamanho REAL contra o `shared_buffers` declarado — sem isto, "exceeds-cache" e so um rotulo.
    tam=$(PGUSER=postgres psql -h /var/run/postgresql -v ON_ERROR_STOP=1 -tAc \
      "SELECT pg_total_relation_size('bench_contention')" 2>&1)
    # Vazio ou nao-numerico significa que a consulta falhou — e um `-lt` contra vazio e FALSO, entao
    # a guarda de regime passaria calada. Tratar aqui e o que a torna guarda.
    case "$tam" in
      ''|*[!0-9]*) echo "FALHA: nao consegui medir o tamanho da tabela (psql disse: $tam)"; exit 1 ;;
    esac
    echo "   tabela: $((tam / 1048576)) MiB contra shared_buffers=$SB"
    if [ "$regime" = "exceeds-cache" ] && [ "$tam" -lt 33554432 ]; then
      echo "FALHA: o dado ($((tam / 1048576)) MiB) NAO excede os 32 MB de cache — o regime seria falso"; exit 1
    fi

    PGUSER=postgres /root/venv/bin/theodb-bench contention --system theodb \
      --table bench_contention --path columnar \
      --readers "$CONT_LEITORES" --writers "$CONT_ESCRITORES" \
      --read-ops 200 --write-ops 200 --regime "$regime"
    echo "=== contencao :: $regime fim rc=$? $(date -Is) ==="
  done
  echo "=== FIM $(date -Is) ==="
  exit 0
fi

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
