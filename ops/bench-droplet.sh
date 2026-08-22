#!/usr/bin/env bash
# Uma corrida ponta a ponta, da maquina de desenvolvimento: cria o host, mede, colhe, DESTROI.
#
# A destruicao e garantida por `trap EXIT`, e nao por uma linha no fim. Medido em 2026-08-21: o
# desperdicio de uma sessao inteira nao veio de droplet caro, veio de droplet OCIOSO — duas mortes
# de script deixaram a maquina de pe sem ninguem medindo, ~70 min a US$ 0,75/h. Um `delete` que
# depende do caminho feliz nao acontece justamente quando mais importa.
set -uo pipefail

SNAPSHOT="${SNAPSHOT:-theo-bench-base}"       # nome do snapshot provisionado; vazio => ubuntu limpa
REGIAO="${REGIAO:-nyc1}"
TAMANHO="${TAMANHO:-g-16vcpu-64gb}"
SSH_KEY="${SSH_KEY:-58598100}"
SUITE="${SUITE:-analytical/crossover/row-count}"
PROFILE="${PROFILE:-research}"
MODE="${MODE:-suite}"
# Fatores de escala do MODE=tpch. Precisa existir AQUI, e nao so no bench-run, porque o
# ambiente nao atravessa o ssh sozinho: toda variavel que o script remoto le tem de ser
# citada na linha de invocacao. Foi assim que uma corrida de contencao declarou 10M e
# rodou 40M — CONT_LINHAS existia dos dois lados e nao era repassada.
SFS="${SFS:-0.01 0.1}"
SMOKE="${SMOKE:-}"
OMNI_IMAGE="${OMNI_IMAGE:-}"
PGVECTOR_IMAGE="${PGVECTOR_IMAGE:-}"
# Parametros do modo contencao. Precisam estar AQUI e serem encaminhados abaixo: o `bench-run.sh` roda
# no host remoto, e uma variavel exportada aqui nao atravessa o `ssh`. MEDIDO em 2026-08-22: declarei
# `CONT_LINHAS=10000000` na linha de comando, ela nao foi encaminhada, e o executor remoto usou o
# default de 40M do proprio arquivo — a medicao rodou com 4x o volume declarado, e a unica pista foi a
# tabela dar 125 MiB onde 10M linhas dao 31 MB. Declarar nao e efetivar.
CONT_LINHAS="${CONT_LINHAS:-10000000}"
CONT_LEITORES="${CONT_LEITORES:-4}"
CONT_ESCRITORES="${CONT_ESCRITORES:-2}"
CPU_SET="${CPU_SET:-}"
MEM_MAX="${MEM_MAX:-}"
# TAGS aceita `nome:ref` — o ref e um commit-ish do repo do theo-db, e a imagem `theodb:nome` e
# construida a partir dele NO HOST. E assim que se compara dois commits: mesma maquina, mesmo dia,
# mesmos parametros, diferindo so no codigo. `nome` sozinho assume que a imagem ja existe.
TAGS="${TAGS:-fix:HEAD}"
DB_REPO="${DB_REPO:-$(cd "$(dirname "$0")/../../theo-db" 2>/dev/null && pwd)}"
BENCH_REPO="${BENCH_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DESTINO="${DESTINO:-./resultados}"
MANTER="${MANTER:-0}"                          # MANTER=1 nao destroi (depuracao); custa US$ 0,75/h
# Teto para a chamada de medicao remota. MEDIDO em 2026-08-21: a corrida terminou no droplet com
# `rc=0` e o SSH que a conduzia morreu sem devolver — o script local ficou pendurado DUAS HORAS
# esperando, com o host ocioso cobrando US$ 0,75/h. O `trap` protege contra o script MORRER; nao
# protege contra ele TRAVAR, e sao falhas diferentes. `ServerAliveInterval` derruba a sessao quando o
# outro lado some; o `timeout` cobre o caso em que ele responde e nunca termina.
MEDICAO_TIMEOUT="${MEDICAO_TIMEOUT:-14400}"   # 4 h: acima do sweep mais longo, muito abaixo de uma noite

# Droplets que NAO sao de medicao e nunca se toca. Guarda explicita, nao confianca no nome que passei.
PROIBIDOS="theo-e2e-runner theokit-website"

ID=""; IP=""; TMP=""; COLHIDO=0; SEM_RESULTADO=0
limpar() {
  local rc=$?
  [ -n "$TMP" ] && [ -d "$TMP" ] && find "$TMP" -mindepth 0 -delete 2>/dev/null

  # COLHER ANTES DE DESTRUIR, em QUALQUER caminho de saida — sucesso, falha, Ctrl-C, ou o script
  # quebrando no meio. MEDIDO em 2026-08-21: a coleta ficava DEPOIS da medicao, o script quebrou
  # entre as duas, e o trap destruiu o droplet com os resultados dentro. A medicao tinha rodado e
  # terminado com rc=0; o que se perdeu foi so a copia. Numa corrida de verdade seria perda
  # permanente de dado pago, e a mensagem "NAO destrua ate copiar" nao adianta quando quem destroi
  # e o proprio trap.
  if [ -n "$IP" ] && [ "$COLHIDO" = "0" ]; then
    echo ">>> colhendo resultados antes de destruir"
    mkdir -p "$DESTINO"
    # So a corrida DESTA execucao. `res-*` varreria tambem o que veio dentro do snapshot.
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$IP" \
      'st=$(cat /root/ULTIMA_CORRIDA 2>/dev/null); [ -n "$st" ] || exit 42
       tar -czf /root/resultados.tgz "/root/res-$st" /root/bench-run.log 2>/dev/null' 2>/dev/null
    [ $? -eq 42 ] && SEM_RESULTADO=1
    if scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -q "root@$IP:/root/resultados.tgz" "$DESTINO/$NOME.tgz" 2>/dev/null \
       && [ -s "$DESTINO/$NOME.tgz" ] && tar -tzf "$DESTINO/$NOME.tgz" >/dev/null 2>&1; then
      COLHIDO=1; echo "    $DESTINO/$NOME.tgz ($(du -h "$DESTINO/$NOME.tgz" | cut -f1))"
    else
      # Copiar NAO e colher. Um `scp` bem-sucedido de um arquivo vazio ou corrompido reportaria
      # sucesso e o droplet seria destruido com o resultado dentro. Verificar o que se afirma ter
      # colhido e a mesma disciplina que o resto deste sistema aplica a medicao.
      echo "!!! NAO consegui colher de $IP (ausente, vazio ou corrompido)"
    fi
  fi

  # Um droplet cujos resultados nao foram colhidos NAO e destruido automaticamente. Ficar de pe
  # cobrando e ruim; destruir dado que custou uma hora de host e pior, e irreversivel.
  #
  # MAS: quando a corrida falhou ANTES de produzir qualquer resultado, nao ha o que perder, e manter
  # a maquina de pe seria so queimar dinheiro. Medido em 2026-08-21: o smoke reprovou, nao havia
  # nenhum `res-*` desta execucao, e a guarda manteve um droplet cobrando por nada. A guarda existe
  # para proteger DADO, nao para reagir a qualquer falha.
  if [ -n "$ID" ] && [ "$COLHIDO" = "0" ] && [ "$SEM_RESULTADO" = "1" ] && [ "$MANTER" != "1" ]; then
    echo ">>> nenhum resultado foi produzido nesta corrida — nada a perder; destruindo"
  elif [ -n "$ID" ] && [ "$COLHIDO" = "0" ] && [ "$MANTER" != "1" ]; then
    echo ">>> droplet $ID ($IP) MANTIDO DE PE: a coleta falhou e destruir perderia o resultado."
    echo "    colha a mao, depois: doctl compute droplet delete $ID --force"
    exit $rc
  fi

  if [ -n "$ID" ]; then
    if [ "$MANTER" = "1" ]; then
      echo ">>> MANTER=1: droplet $ID ($IP) DE PE, cobrando US\$ 0,75/h. Destrua com:"
      echo "    doctl compute droplet delete $ID --force"
    else
      echo ">>> destruindo droplet $ID"
      doctl compute droplet delete "$ID" --force || echo "!!! FALHA AO DESTRUIR $ID — destrua a mao"
    fi
  fi
  exit $rc
}
trap limpar EXIT INT TERM

NOME="theo-bench-$(date -u +%Y%m%dT%H%M%SZ)"
for p in $PROIBIDOS; do
  [ "$NOME" = "$p" ] && { echo "recusado: nome colide com droplet protegido"; exit 1; }
done

echo "=== criando $NOME ($TAMANHO, $REGIAO) $(date -Is) ==="
IMAGEM="ubuntu-24-04-x64"
if [ -n "$SNAPSHOT" ]; then
  SNAP_ID=$(doctl compute snapshot list --format ID,Name --no-header 2>/dev/null | awk -v n="$SNAPSHOT" '$2==n{print $1; exit}')
  if [ -n "$SNAP_ID" ]; then IMAGEM="$SNAP_ID"; echo "    a partir do snapshot $SNAPSHOT ($SNAP_ID)"
  else echo "    snapshot '$SNAPSHOT' nao encontrado — usando ubuntu limpa (provision.sh roda inteiro)"; fi
fi

ID=$(doctl compute droplet create "$NOME" --region "$REGIAO" --size "$TAMANHO" --image "$IMAGEM" \
      --ssh-keys "$SSH_KEY" --tag-names theo-test,ephemeral --wait --format ID --no-header) || exit 1
IP=$(doctl compute droplet get "$ID" --format PublicIPv4 --no-header)
echo "    id=$ID ip=$IP"

echo "=== aguardando ssh ==="
for _ in $(seq 1 60); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes "root@$IP" true 2>/dev/null && break
  sleep 5
done
ssh -o StrictHostKeyChecking=no "root@$IP" true || { echo "FALHA: ssh nunca respondeu"; exit 1; }

echo "=== enviando codigo e provisionando ==="
scp -o StrictHostKeyChecking=no -q "$(dirname "$0")/provision.sh" "$(dirname "$0")/bench-run.sh" "root@$IP:/root/"
ssh -o StrictHostKeyChecking=no "root@$IP" 'chmod +x /root/provision.sh /root/bench-run.sh'

# O ARNES precisa chegar antes do provisionamento: `provision.sh` cria o venv A PARTIR de /root/bench,
# em modo editavel, porque so assim `schemas/` fica ao lado do pacote. Medido: sem este envio, um host
# limpo NUNCA passa no portao — foi o defeito que o primeiro teste ponta a ponta encontrou, e que
# nenhuma checagem de sintaxe encontraria.
# BUNDLE, e nao tarball. MEDIDO: o arnes valida `clean_source_tree` rodando `git status --porcelain`
# na arvore DO PROPRIO ARNES (o campo e `benchmark_dirty`, e a descricao do portao diz "Benchmark
# source tree was committed"). Um tarball nao carrega `.git`, entao `git status` falha, o portao fica
# UNAVAILABLE e — no perfil `release`, onde ele e obrigatorio — invalida a corrida.
#
# Um bundle resolve sem credencial e sem rede: e um repositorio git completo num arquivo. O host
# clona dele e fica com arvore limpa num SHA conhecido, que e exatamente o que o portao pede.
TMP="$(mktemp -d)"
git -C "$BENCH_REPO" bundle create "$TMP/bench.bundle" HEAD --branches 2>/dev/null \
  || git -C "$BENCH_REPO" bundle create "$TMP/bench.bundle" HEAD \
  || { echo "FALHA: git bundle do arnes"; exit 1; }
BENCH_SHA="$(git -C "$BENCH_REPO" rev-parse --short HEAD)"
scp -o StrictHostKeyChecking=no -q "$TMP/bench.bundle" "root@$IP:/root/"
ssh -o StrictHostKeyChecking=no "root@$IP" \
  'rm -rf /root/bench 2>/dev/null; git clone -q /root/bench.bundle /root/bench 2>&1 | tail -2
   git -C /root/bench status --porcelain | head -3' \
  || { echo "FALHA: clonar o arnes do bundle"; exit 1; }
echo "    arnes clonado em $BENCH_SHA (arvore git de verdade, nao tarball)"

# O portao PRIMEIRO. Se o snapshot envelheceu, descobre-se aqui — em segundos — e nao depois de
# compilar. Se reprovar, provisiona e verifica de novo; se reprovar outra vez, para.
if ! ssh -o StrictHostKeyChecking=no "root@$IP" 'bash /root/provision.sh --verify'; then
  echo "=== portao reprovou: provisionando ==="
  ssh -o StrictHostKeyChecking=no "root@$IP" 'bash /root/provision.sh' || { echo "FALHA: provisionamento"; exit 1; }
  ssh -o StrictHostKeyChecking=no "root@$IP" 'bash /root/provision.sh --verify' || { echo "FALHA: portao reprovou apos provisionar"; exit 1; }
fi

# --- imagens: uma por tag, construida no host a partir do ref -------------------------------
NOMES=""
for spec in $TAGS; do
  nome="${spec%%:*}"; ref="${spec#*:}"
  NOMES="$NOMES $nome"
  if [ "$nome" = "$ref" ]; then
    echo "=== imagem theodb:$nome — assumida presente (nenhum ref dado) ==="
    ssh -o StrictHostKeyChecking=no "root@$IP" "docker image inspect theodb:$nome >/dev/null 2>&1" \
      || { echo "FALHA: theodb:$nome nao existe no host e nenhum ref foi dado"; exit 1; }
    continue
  fi
  if ssh -o StrictHostKeyChecking=no "root@$IP" "docker image inspect theodb:$nome >/dev/null 2>&1"; then
    echo "=== imagem theodb:$nome ja existe — reusando ==="
    continue
  fi
  sha="$(git -C "$DB_REPO" rev-parse --short "$ref")" || { echo "FALHA: ref '$ref' nao resolve"; exit 1; }
  echo "=== construindo theodb:$nome a partir de $ref ($sha) $(date -Is) ==="
  git -C "$DB_REPO" archive "$ref" -o "$TMP/$nome.tar" || { echo "FALHA: git archive $ref"; exit 1; }
  scp -o StrictHostKeyChecking=no -q "$TMP/$nome.tar" "root@$IP:/root/"
  ssh -o StrictHostKeyChecking=no "root@$IP" "docker build -t theodb:$nome - < /root/$nome.tar" \
    || { echo "FALHA: build de theodb:$nome"; exit 1; }
  echo "=== theodb:$nome pronto (de $sha) $(date -Is) ==="
done
TAGS="${NOMES# }"

# PORTAO: toda variavel que o bench-run.sh le tem de aparecer na invocacao remota.
#
# Manter isto como lista na cabeca ja falhou uma vez — CONT_LINHAS existia dos dois lados,
# nao era repassada, e uma corrida declarou 10M enquanto rodava 40M. A lista nao e o
# conserto; o portao e. Ele le o proprio bench-run.sh e compara.
FALTANDO=$(python3 - "$(dirname "$0")/bench-run.sh" "$0" <<'PYEOF'
import re, sys
lidas = set(re.findall(r'^(\w+)="\$\{\1:-', open(sys.argv[1], encoding="utf-8").read(), re.M))
linhas = [l for l in open(sys.argv[2], encoding="utf-8") if "/root/bench-run.sh" in l]
linha = max(linhas, key=len) if linhas else ""
passadas = set(re.findall(r"(\w+)='\$", linha))
print(" ".join(sorted(lidas - passadas)))
PYEOF
)
if [ -n "$FALTANDO" ]; then
  echo "FALHA: o bench-run.sh le variaveis que a invocacao remota nao repassa: $FALTANDO"
  echo "       elas ficariam no default silenciosamente, e a corrida mediria outra coisa."
  exit 1
fi

echo "=== medindo (suite=$SUITE tags=$TAGS) ==="
timeout "$MEDICAO_TIMEOUT" ssh -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  "root@$IP" "SUITE='$SUITE' TAGS='$TAGS' PROFILE='$PROFILE' CPU_SET='$CPU_SET' MEM_MAX='$MEM_MAX' MODE='$MODE' CONT_LINHAS='$CONT_LINHAS' CONT_LEITORES='$CONT_LEITORES' CONT_ESCRITORES='$CONT_ESCRITORES' SFS='$SFS' SMOKE='$SMOKE' OMNI_IMAGE='$OMNI_IMAGE' PGVECTOR_IMAGE='$PGVECTOR_IMAGE' /root/bench-run.sh"
RC=$?
# 124 e o codigo do `timeout`. Dizer isso em vez de deixar um rc=124 solto importa: a corrida pode ter
# TERMINADO no droplet e so a conducao ter travado — foi o que aconteceu — e nesse caso os resultados
# existem e a coleta abaixo os traz.
[ "$RC" -eq 124 ] && echo "!!! a conducao da medicao estourou ${MEDICAO_TIMEOUT}s; os resultados podem existir no host — a coleta segue"

echo "=== fim rc=$RC $(date -Is) ==="
exit $RC
