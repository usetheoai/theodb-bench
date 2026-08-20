#!/usr/bin/env bash
# changelog-section.sh — imprime a seção de UMA versão do CHANGELOG, sem o cabeçalho.
#
# POR QUE EXISTE. Nenhum workflow deste repositório criava release (medido 2026-08-20: zero
# ocorrências de `gh release`, `action-gh-release` ou gatilho `release:` nos nove arquivos), e
# `main` ficou em 0.158.0 desde 2026-07-29 com DUAS versões escritas no changelog e nunca cortadas.
#
# POR QUE NÃO `gh release --generate-notes`. Ele deriva das mensagens de commit. A Regra 6 deste
# projeto diz que o CHANGELOG é o contrato com quem consome, escrito para humanos — usar o log
# geraria uma nota que ninguém revisou, ao lado de uma que alguém escreveu.
#
# FALHA ALTO E CLARO (rules/error-handling.md § 2): versão ausente do CHANGELOG é erro com a lista
# do que existe, nunca nota vazia. Uma release publicada com corpo vazio é pior que nenhuma — ela
# afirma que não houve mudança.
#
# Uso:  scripts/changelog-section.sh 0.160.1 [CHANGELOG.md]
set -euo pipefail

VERSION="${1:?uso: changelog-section.sh <versao-sem-v> [arquivo]}"
FILE="${2:-CHANGELOG.md}"
VERSION="${VERSION#v}"

[ -f "$FILE" ] || { echo "changelog-section: $FILE não existe" >&2; exit 2; }

SECTION=$(awk -v v="$VERSION" '
  $0 ~ "^## \\[" v "\\]" { found=1; next }
  found && /^## \[/      { exit }
  found                  { print }
' "$FILE")

if [ -z "${SECTION//[[:space:]]/}" ]; then
  echo "changelog-section: nenhuma seção '## [$VERSION]' com conteúdo em $FILE." >&2
  echo "Versões presentes:" >&2
  grep -oE '^## \[[^]]+\]' "$FILE" | head -10 >&2
  exit 1
fi

printf '%s\n' "$SECTION"
