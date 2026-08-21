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
TAGS="${TAGS:-fix}"
DESTINO="${DESTINO:-./resultados}"
MANTER="${MANTER:-0}"                          # MANTER=1 nao destroi (depuracao); custa US$ 0,75/h

# Droplets que NAO sao de medicao e nunca se toca. Guarda explicita, nao confianca no nome que passei.
PROIBIDOS="theo-e2e-runner theokit-website"

ID=""; IP=""
limpar() {
  local rc=$?
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

# O portao PRIMEIRO. Se o snapshot envelheceu, descobre-se aqui — em segundos — e nao depois de
# compilar. Se reprovar, provisiona e verifica de novo; se reprovar outra vez, para.
if ! ssh -o StrictHostKeyChecking=no "root@$IP" 'bash /root/provision.sh --verify'; then
  echo "=== portao reprovou: provisionando ==="
  ssh -o StrictHostKeyChecking=no "root@$IP" 'bash /root/provision.sh' || { echo "FALHA: provisionamento"; exit 1; }
  ssh -o StrictHostKeyChecking=no "root@$IP" 'bash /root/provision.sh --verify' || { echo "FALHA: portao reprovou apos provisionar"; exit 1; }
fi

echo "=== medindo (suite=$SUITE tags='$TAGS') ==="
ssh -o StrictHostKeyChecking=no "root@$IP" "SUITE='$SUITE' TAGS='$TAGS' /root/bench-run.sh"
RC=$?

echo "=== colhendo resultados ==="
mkdir -p "$DESTINO"
ssh -o StrictHostKeyChecking=no "root@$IP" 'tar -czf /root/resultados.tgz /root/res-* /root/bench-run.log 2>/dev/null; true'
scp -o StrictHostKeyChecking=no -q "root@$IP:/root/resultados.tgz" "$DESTINO/$NOME.tgz" \
  && echo "    $DESTINO/$NOME.tgz" || echo "!!! nao consegui colher — NAO destrua ate copiar"

echo "=== fim rc=$RC $(date -Is) ==="
exit $RC
