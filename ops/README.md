# ops — provisionar e executar num host de bench

Dois arquivos, com responsabilidades que não se misturam.

| arquivo | responsabilidade | roda onde |
|---|---|---|
| `provision.sh` | **o que a máquina precisa ter** — fonte de verdade | host de bench, uma vez |
| `bench-run.sh` | **como uma medição é executada** — portão, smoke, sweep | host de bench, por corrida |

## Por que um script e não só um snapshot

Um snapshot DigitalOcean é rápido e **opaco**: em três meses ninguém sabe o que tem dentro. Este
projeto se apoia no princípio oposto — até `shared_buffers` é declarado, nunca default, porque um
artefato com estado não-declarado não é reproduzível.

Então: **`provision.sh` é a verdade; o snapshot é cache derivado dele.** Se o snapshot sumir ou
envelhecer, roda-se o script e ele volta.

```bash
./provision.sh            # provisiona do zero (~40 s + venv)
./provision.sh --verify   # só verifica; idempotente, barato, seguro a qualquer momento
```

## O portão, e por que ele existe

`--verify` checa cada capacidade que **já custou uma corrida perdida** em 2026-08-21:

| capacidade | o que a ausência causa |
|---|---|
| `buildx` | `COPY <<EOF` falha no **passo 26 de 28**, depois de compilar a extensão — 40 min perdidos |
| `psycopg` (extra `[postgres]`) | adapter recusa no bootstrap |
| `schemas/` ao lado do pacote | arnês invalida o bundle **no fim** da corrida |

As três têm a mesma forma: capacidade presente na máquina de desenvolvimento, ausente num host
limpo, descoberta **depois** do trabalho caro. Falhar no início custa segundos; falhar no passo 26
custa a compilação inteira.

Por isso `pip install -e` e não `pip install`: editável mantém o pacote enraizado no checkout, onde
`schemas/` existe. Um install de tarball perde esse diretório silenciosamente.

## Executar

```bash
SUITE=analytical/crossover/row-count TAGS="base fix" ./bench-run.sh
```

Ordem imposta pelo executor, e nenhuma etapa é opcional:

1. **portão** — todas as capacidades, antes de qualquer trabalho caro
2. **servidor** — sobe com os `-c` declarados, e cria `theodb-bench-parquet` **dentro** do contêiner,
   pertencendo ao usuário do banco (sem ele o arnês reporta `sut_alive` FAIL e culpa o servidor por
   uma falha que foi de uma consulta)
3. **proveniência** — versão e `extversion` **lidas do servidor**, nunca da tag da imagem (B-069)
4. **smoke** — `analytical/synthetic/paths`, um N, três caminhos, ~35 s; reprova aqui e o sweep caro
   não roda
5. **sweep** — a suíte de verdade, uma vez por tag

A regra que organiza os dois arquivos: **o que MEDE aborta em erro; o que apenas REGISTRA nunca
aborta.** Uma linha de proveniência com nome de extensão obsoleto, sob `set -e`, já matou uma corrida
e custou 29 minutos de host pago.

## Teto de veredito

O arnês valida `clean_source_tree`. Código enviado por tarball reprova esse item, e o veredito fica
em `EXPLORATORY` — **`release` exige `git clone` num SHA conhecido**. Ver
`theo-db/wiki/runbooks/droplet-de-medicao.md § Tarball ou git clone`.
