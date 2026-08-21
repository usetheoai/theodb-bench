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
# TAGS aceita `nome:ref` — o ref e um commit-ish do repo do theo-db, e a imagem `theodb:nome` e
# construida a partir dele NO HOST. E assim que se compara dois commits: mesma maquina, mesmo dia,
# mesmos parametros, diferindo so no codigo. `nome` sozinho assume que a imagem ja existe.
TAGS="${TAGS:-fix:HEAD}"
DB_REPO="${DB_REPO:-$(cd "$(dirname "$0")/../../theo-db" 2>/dev/null && pwd)}"
BENCH_REPO="${BENCH_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DESTINO="${DESTINO:-./resultados}"
MANTER="${MANTER:-0}"                          # MANTER=1 nao destroi (depuracao); custa US$ 0,75/h

# Droplets que NAO sao de medicao e nunca se toca. Guarda explicita, nao confianca no nome que passei.
PROIBIDOS="theo-e2e-runner theokit-website"

ID=""; IP=""; TMP=""; COLHIDO=0
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
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$IP" \
      'tar -czf /root/resultados.tgz /root/res-* /root/bench-run.log 2>/dev/null; true' 2>/dev/null
    if scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -q "root@$IP:/root/resultados.tgz" "$DESTINO/$NOME.tgz" 2>/dev/null; then
      COLHIDO=1; echo "    $DESTINO/$NOME.tgz"
    else
      echo "!!! NAO consegui colher de $IP"
    fi
  fi

  # Um droplet cujos resultados nao foram colhidos NAO e destruido automaticamente. Ficar de pe
  # cobrando e ruim; destruir dado que custou uma hora de host e pior, e irreversivel.
  if [ -n "$ID" ] && [ "$COLHIDO" = "0" ] && [ "$MANTER" != "1" ]; then
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
TMP="$(mktemp -d)"
git -C "$BENCH_REPO" archive HEAD -o "$TMP/bench.tar" || { echo "FALHA: git archive do arnes"; exit 1; }
scp -o StrictHostKeyChecking=no -q "$TMP/bench.tar" "root@$IP:/root/"
ssh -o StrictHostKeyChecking=no "root@$IP" 'mkdir -p /root/bench && tar -xf /root/bench.tar -C /root/bench' \
  || { echo "FALHA: extrair o arnes"; exit 1; }

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

echo "=== medindo (suite=$SUITE tags=$TAGS) ==="
ssh -o StrictHostKeyChecking=no "root@$IP" "SUITE='$SUITE' TAGS='$TAGS' /root/bench-run.sh"
RC=$?

echo "=== fim rc=$RC $(date -Is) ==="
exit $RC
