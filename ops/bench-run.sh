#!/usr/bin/env bash
# Executor de medicao. Roda no HOST DE BENCH, nao na maquina de desenvolvimento.
#
# Regra que organiza o arquivo inteiro, e que custou uma sessao inteira para ser escrita:
#   o que MEDE aborta em erro; o que apenas REGISTRA nunca aborta.
# Toda morte prematura desta sessao foi um comando de registro derrubando a corrida.
set -uo pipefail

SUITE="${SUITE:-analytical/crossover/row-count}"
PROFILE="${PROFILE:-research}"
# Repeticoes por ponto. Vazio = o `default_repetitions` da suite decide, que e o
# comportamento que existia. Precisa ser exposto porque o perfil `release` — o UNICO que o
# projeto define como publicavel — exige >= 5, e nao havia por onde pedir: medido no acervo em
# 2026-08-22, 18 bundles, 15 `research`, 3 `nightly`, ZERO `release`.
REPS="${REPS:-}"
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
# O marcador de coleta e escrito AQUI, no comeco — e nao so no fim de um modo bem-sucedido.
#
# `bench-droplet.sh` colhe lendo este arquivo; sem ele, a coleta sai com 42 e a corrida inteira e
# descartada como "sem resultado". MEDIDO em 2026-08-23: o regime `memory-resident` da contencao
# produziu numeros e terminou rc=0, o regime seguinte foi recusado por um portao, o marcador nunca
# foi escrito, e os numeros do primeiro foram jogados fora junto com o droplet.
#
# O comentario da guarda no bench-droplet diz "a guarda existe para proteger DADO, nao para reagir a
# qualquer falha" — e escrevendo o marcador so no caminho feliz ela fazia exatamente o contrario.
# Resultado parcial e resultado.
echo "$STAMP" > /root/ULTIMA_CORRIDA

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

# `THEODB_ADMIT_TRACE=1` faz o motor DIZER por que recusou cada candidato ao agregado colunar
# (`am/columnar_agg.rs`), em vez de deixar a recusa invisivel no plano. Precisa chegar ao SERVIDOR
# e nao ao cliente: quem emite o aviso e o backend.
#
# Vazio por default: o proprio codigo avisa que a resolucao da variavel cai no caminho quente do
# planner, e ligar isso numa corrida de medicao mediria o trace junto.
ADMIT_TRACE="${ADMIT_TRACE:-}"
# `PERF=1` liga os contadores de hardware (ciclos, instrucoes, cache) via `perf stat`.
#
# Opt-in e nao default, e a razao e medida: o coletor anexa um `perf stat` ao processo, e
# isso custa. Mas ate 2026-08-23 ele estava DESLIGADO EM TODA CORRIDA — medido no acervo,
# 0 de 18 bundles tem `perf.cycles`. Duas causas empilhadas: o coletor deduzia a capacidade
# de `perf_event_paranoid` (consertado — root contorna a politica), e nenhum script passava
# `--perf`. Consertar a primeira sem a segunda nao mudaria nada.
PERF="${PERF:-}"
# Sistema medido pelo `MODE=suite`. Ele cravava `--system theodb`, e com isso TRES suites do
# registro — `vector/sift/scann-ah`, `vector/sift1m/scann-ah`, `vector/synthetic/scann-sweep` — eram
# ESTRUTURALMENTE inalcancaveis: elas medem o access method `scann` do AlloyDB Omni, que o TheoDB nao
# tem. MEDIDO em 2026-08-23: rodei a varredura do `pre_reordering` e os nove pontos vieram
# `not measured`, com o bundle nomeado `...-scann-ah-theodb-...`. O arnes nao errou; ele mediu
# exatamente o que lhe foi pedido, contra o sistema errado.
SISTEMA="${SISTEMA:-theodb}"
# O endereco acompanha o sistema. `theodb` roda no socket local; qualquer externo roda no conteiner
# que `subir_externo` levanta na 55460.
#
# Por VARIAVEL DE AMBIENTE, e nao por flag: o subcomando `run` NAO tem `--dsn` (so `tpch` e
# `contention` tem), e o adapter cai em `postgresql:///postgres` — URI de host vazio, que o libpq
# resolve por `PGHOST`/`PGPORT`. Usar o mecanismo que a plataforma ja oferece e o degrau 3 da
# parsimony ladder; adicionar uma flag ao CLI seria o degrau 6 para o mesmo efeito.
#
# MEDIDO em 2026-08-23: a tentativa com `--dsn` morreu em `unrecognized arguments` e custou um droplet.
PG_EXTERNO=""
if [ "$SISTEMA" != "theodb" ]; then
  PG_EXTERNO="PGHOST=127.0.0.1 PGPORT=55460 PGUSER=postgres PGPASSWORD=x"
fi


subir() {
  local tag="$1"
  docker rm -f theodb >/dev/null 2>&1 || true
  docker run -d --name theodb -e POSTGRES_HOST_AUTH_METHOD=trust \
    ${ADMIT_TRACE:+-e THEODB_ADMIT_TRACE=1} \
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

# Qual dataset a suite EXIGE, perguntado ao registro em vez de adivinhado pelo nome.
# Vazio = suite sintetica, que nao exige nada.
dataset_exigido() {
  /root/venv/bin/python -c "
import sys
from theodb_bench.registry import BENCHMARKS
e = BENCHMARKS.get(sys.argv[1])
print((e.requires_dataset or '') if e else '')
" "$1" 2>/dev/null
}

medir() {
  local tag="$1" suite="$2" saida="$3"
  echo "=== $tag :: $suite inicio $(date -Is) ==="
  # Uma suite nomeada por dataset mede o corpus SINTETICO se ninguem passar `--dataset` —
  # e o bundle sai com o nome do dataset no id e sem o bloco que o identifica. O preflight
  # do arnes agora recusa isso, entao aqui e onde o dado tem de chegar.
  local ds; ds="$(dataset_exigido "$suite")"
  local arg_ds=""
  if [ -n "$ds" ]; then
    echo "-- suite exige dataset '$ds'; buscando e verificando --"
    /root/venv/bin/theodb-bench dataset fetch "$ds" || { echo "FALHA: fetch de $ds"; return 1; }
    arg_ds="--dataset $ds"
  fi
  # Sob MEM_MAX o arnes roda DENTRO de um cgroup com limite, porque e o que ele exige para marcar
  # `memory_limit` como respeitado — aplicar o limite ele mesmo pediria privilegio e teria efeito
  # colateral sobre o host, entao ele LE o limite que ja vale. `systemd-run --scope` e o mecanismo
  # nativo para criar esse cgroup (degrau 3 da parsimony ladder), e sem ele os perfis `nightly` e
  # `release` sao inalcancaveis.
  if [ -n "$MEM_MAX" ] && command -v systemd-run >/dev/null 2>&1; then
    env ${PG_EXTERNO:-PGUSER=postgres} systemd-run --scope --quiet -p "MemoryMax=$MEM_MAX" \
      /root/venv/bin/theodb-bench run "$suite" \
      --system "$SISTEMA" --profile "$PROFILE" --output "$saida" $arg_ds \
      ${REPS:+--repetitions "$REPS"} ${PERF:+--perf} \
      ${CPU_SET:+--cpu-set "$CPU_SET"} --memory "$MEM_MAX"
  else
    env ${PG_EXTERNO:-PGUSER=postgres} /root/venv/bin/theodb-bench run "$suite" \
      --system "$SISTEMA" --profile "$PROFILE" --output "$saida" $arg_ds \
      ${REPS:+--repetitions "$REPS"} ${PERF:+--perf} \
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

# ------------------------------------------------- casado por recall, DUAS familias (B-057)
#
# POR QUE ESTE MODO EXISTE, e por que o `headtohead` acima nao serve para isto. Aquele roda
# UMA suite contra tres sistemas, que e o certo quando os tres aceitam os mesmos botoes. Nao
# e o caso entre familias de indice: o nosso e o do pgvector sao GRAFO e tomam `ef_search`;
# o do AlloyDB Omni e ARVORE e toma `num_leaves_to_search`. Medido em 2026-08-22, ao rodar
# `vector/sift/hnsw` contra os tres: a perna do Omni voltou `run_not_refused` — o arnes
# recusou-se a medir, corretamente, porque nao sabe aplicar o botao pedido.
#
# O comando `head2head` do arnes JA tem a forma certa (`--benchmark-a` e `--benchmark-b`,
# uma suite por sistema) e eu construi o modo errado sem olhar. Este modo so o invoca.
if [ "$MODE" = "recall-casado" ]; then
  SUITE_A="${SUITE_A:-vector/sift/hnsw}"
  SUITE_B="${SUITE_B:-vector/sift/scann-ah}"
  SIST_A="${SIST_A:-theodb}"
  SIST_B="${SIST_B:-alloydbomni}"

  docker pull "$OMNI_IMAGE" >/dev/null 2>&1 || { echo "FALHA: pull do Omni"; exit 1; }
  subir "${TAGS%% *}" || exit 1
  subir_externo omni "$OMNI_IMAGE" 55460 || exit 1

  echo "-- proveniencia, lida de cada servidor --"
  PGUSER=postgres psql -h /var/run/postgresql -tAc "select 'theodb: '||version()" 2>&1 | head -1 || true
  PGPASSWORD=x psql -h 127.0.0.1 -p 55460 -U postgres -tAc "select 'omni:   '||version()" 2>&1 | head -1 || true

  ARG_DS=""
  DS="$(dataset_exigido "$SUITE_A")"
  if [ -n "$DS" ]; then
    echo "-- suite exige dataset '$DS'; buscando e verificando --"
    /root/venv/bin/theodb-bench dataset fetch "$DS" || { echo "FALHA: fetch de $DS"; exit 1; }
    ARG_DS="--dataset $DS"
  fi

  mkdir -p "/root/res-$STAMP"
  echo "=== recall-casado $SIST_A($SUITE_A) x $SIST_B($SUITE_B) inicio $(date -Is) ==="
  /root/venv/bin/theodb-bench head2head \
    --system-a "$SIST_A" --dsn-a "postgresql:///postgres?host=/var/run/postgresql&user=postgres" \
    --benchmark-a "$SUITE_A" \
    --system-b "$SIST_B" --dsn-b "postgresql://postgres:x@127.0.0.1:55460/postgres" \
    --benchmark-b "$SUITE_B" \
    $ARG_DS | tee "/root/res-$STAMP/recall-casado.txt"
  echo "=== recall-casado fim rc=$? $(date -Is) ==="

  echo "$STAMP" > /root/ULTIMA_CORRIDA
  echo "=== FIM $(date -Is) resultados em /root/res-$STAMP ==="
  touch /root/PRONTO
  exit 0
fi

# ---------------------------------------------------------------- tres vias (B-059 bullet 4)
if [ "$MODE" = "headtohead" ]; then
  # Perfil que exige isolamento sem isolamento declarado produz bundle INVALID depois de
  # medir tudo — medido em 2026-08-22: as tres pernas rodaram, deram numero, e sairam
  # `INVALID: cpu_limit, memory_limit`. O caminho de tres vias tem laco proprio e nao
  # passava por `medir()`, entao os flags que eu tinha acrescentado la nao chegavam aqui.
  # Recusar ANTES de medir custa um segundo; descobrir depois custou a corrida inteira.
  case "$PROFILE" in
    pr|nightly|release)
      [ -n "$CPU_SET" ] && [ -n "$MEM_MAX" ] || {
        echo "FALHA: perfil '$PROFILE' exige isolamento e CPU_SET/MEM_MAX nao foram dados;"
        echo "       as tres pernas mediriam e o bundle sairia INVALID no fim."
        exit 1
      } ;;
  esac

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

  # Mesmo contrato do `medir`: se a suite nomeia um dataset, ele e buscado e passado.
  ARG_DS=""
  DS_H2H="$(dataset_exigido "$SUITE")"
  if [ -n "$DS_H2H" ]; then
    echo "-- suite exige dataset '$DS_H2H'; buscando e verificando --"
    /root/venv/bin/theodb-bench dataset fetch "$DS_H2H" || { echo "FALHA: fetch de $DS_H2H"; exit 1; }
    ARG_DS="--dataset $DS_H2H"
  fi

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
        $ARG_DS ${REPS:+--repetitions "$REPS"} ${PERF:+--perf} \
        ${CPU_SET:+--cpu-set "$CPU_SET"} ${MEM_MAX:+--memory "$MEM_MAX"} \
        --output "/root/res-$STAMP/$sist"
    else
      PGUSER=postgres /root/venv/bin/theodb-bench run "$SUITE" --system "$sist" \
        --profile "$PROFILE" $ARG_DS ${REPS:+--repetitions "$REPS"} ${PERF:+--perf} \
        ${CPU_SET:+--cpu-set "$CPU_SET"} ${MEM_MAX:+--memory "$MEM_MAX"} \
        --output "/root/res-$STAMP/$sist"
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

    # `tee`, e nao so stdout. Medido em 2026-08-22: os dois regimes mediram, imprimiram o
    # JSON, e a coleta do `bench-droplet.sh` reportou "nenhum resultado foi produzido" e
    # destruiu a maquina — CERTO, sobre uma ausencia real, porque nada tinha sido escrito em
    # /root/res-$STAMP. Os numeros so sobreviveram porque estavam no log local de quem
    # lancou, e tiveram de ser recortados dele.
    #
    # Isto NAO transforma a contencao numa suite: ela continua sem bundle validado, e o
    # [[B-104]] registra isso. O que muda e que o resultado passa a ser COLHIDO.
    mkdir -p "/root/res-$STAMP"
    PGUSER=postgres /root/venv/bin/theodb-bench contention --system theodb \
      --table bench_contention --path columnar \
      --readers "$CONT_LEITORES" --writers "$CONT_ESCRITORES" \
      --read-ops 200 --write-ops 200 --regime "$regime" \
      | tee "/root/res-$STAMP/contencao-$regime.json"
    echo "=== contencao :: $regime fim rc=${PIPESTATUS[0]} $(date -Is) ==="
  done
  echo "$STAMP" > /root/ULTIMA_CORRIDA
  echo "=== FIM $(date -Is) resultados em /root/res-$STAMP ==="
  touch /root/PRONTO
  exit 0
fi

# SMOKE PRIMEIRO. Exercita heap+colunar+parquet+oraculo num unico N, em poucos minutos. Se o
# pipeline estiver quebrado, descobre-se aqui — e nao depois de carregar 2 milhoes de linhas
# seis vezes, duas vezes.
PRIMEIRA="${TAGS%% *}"
subir "$PRIMEIRA" || exit 1

# Sistema externo: sobe o conteiner ANTES de medir, do mesmo jeito que os modos comparativos ja
# faziam. Sem isto, `--system alloydbomni` mede contra um servidor que nao existe.
if [ "$SISTEMA" != "theodb" ]; then
  # A imagem depende do sistema: `alloydbomni` traz o `scann`; `pgvector` e a referencia SOTA do
  # nosso proprio `hnsw` — mesma familia de indice, que e o que o ADR-0033 nomeia como a meta
  # ("paridade vetorial classe-pgvector"). Comparar grafo com quantizador mede trade-off, nao paridade.
  case "$SISTEMA" in
    pgvector) IMG_EXT="$PGVECTOR_IMAGE" ;;
    *)        IMG_EXT="$OMNI_IMAGE" ;;
  esac
  docker pull "$IMG_EXT" >/dev/null 2>&1 || { echo "FALHA: pull de $IMG_EXT"; exit 1; }
  subir_externo ext "$IMG_EXT" 55460 || exit 1
  PGPASSWORD=x psql -h 127.0.0.1 -p 55460 -U postgres -tAc "select '$SISTEMA: '||version()" 2>&1 | head -1 || true
fi

# O smoke exercita heap+colunar+parquet+oraculo do TheoDB. Contra um sistema externo ele reprovaria
# por motivo CERTO (o Omni nao tem `theodb_columnar`) e derrubaria a corrida antes do que interessa.
if [ "$SISTEMA" = "theodb" ]; then
  if ! medir "$PRIMEIRA" "$SMOKE" "/root/res-$STAMP/smoke"; then
    echo "=== SMOKE REPROVOU — o sweep caro NAO foi executado ==="
    exit 1
  fi
  echo "=== smoke ok $(date -Is) ==="
else
  echo "=== smoke PULADO: mede o caminho colunar do TheoDB, e o sistema medido e '$SISTEMA' ==="
fi

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
