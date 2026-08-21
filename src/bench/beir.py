"""Corpus real com julgamentos humanos — o que o corpus sintético do `retrieval` não é.

O `retrieval.py` traz a pipeline inteira: nDCG@10, Recall@k, MRR, e as quatro pernas (lexical,
densa, híbrida RRF, híbrida+rerank) sobre o mesmo corpus. **Nenhum benchmark registrado a usava**, e
o único corpus era o `generate_corpus` sintético — cuja própria docstring diz que "exercita a
pipeline e as métricas", sem ser alegação de qualidade.

O efeito disso foi concreto: todo número lexical que este projeto publicou saiu de script ad-hoc,
e o
`m186` chegou a atribuir ao PRODUTO um limite que era do script. Este módulo fecha essa lacuna com
um
corpus que tem **julgamento humano**.

# O que este módulo NÃO faz, e é deliberado

**Não inventa vetores.** O BEIR entrega texto e qrels; embeddings ele não entrega, e produzi-los
exige
um modelo externo. Preencher `Document.vector` com ruído faria a perna densa e a híbrida rodarem e
**parecerem medidas** — números com a aparência de resultado e sem a propriedade. Esse é
exatamente o
modo de falha que a wiki registra em `b018` (3000 vetores idênticos por uma subconsulta não
correlacionada), e ele custou meio dia.

Então o corpus carregado aqui serve a perna **lexical**, e as demais permanecem indisponíveis até
haver fonte de embedding declarada. `load_beir` devolve os vetores como um array de largura zero,
e a
`RetrievalWorkload` que o consome declara `pipelines=("lexical",)` — quem tentar a densa recebe
erro,
não um número.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from theodb_bench.adapters.base import Document
from theodb_bench.bench.retrieval import QuerySet
from theodb_bench.errors import DatasetError, ErrorContext, Phase

#: BEIR identifica documentos e consultas por STRING (`"4983"`, `"query-1"`), e o arnês por `int`.
#: O mapeamento é por posição de leitura e não por `int(id)`: nem todo id do BEIR é numérico, e
#: converter direto quebraria em qualquer corpus cujo id tenha prefixo. Medido no SciFact: os ids
#: são
#: numéricos, mas o `nfcorpus` e o `trec-covid` não são — a conversão direta funcionaria aqui e
#: falharia lá, que é a pior forma de errar.
#: Largura 1, e nao 0, porque `vector(0)` e ilegal no PostgreSQL — medido:
#: `InvalidParameterValue: dimensions for type vector must be at least 1`. A coluna existe porque a
#: forma da tabela de documentos a exige, e carrega um unico zero. Ela NAO e um embedding, e a perna
#: densa e recusada em `cli.py` antes de qualquer consulta — a restricao e imposta, nao so escrita.
_SEM_VETOR_DIM = 1


@contextmanager
def _abrir(raiz: Path, relativo: str) -> Iterator[io.TextIOBase]:
    """Abre um membro do corpus, esteja ele num diretório ou dentro do `.zip` publicado.

    Ler o zip direto, com a `zipfile` da stdlib, evita acrescentar extração à camada de datasets —
    que
    é compartilhada — e evita uma segunda cópia em disco. O manifesto então verifica o sha256 do
    ARQUIVO PUBLICADO, que é garantia mais forte do que verificar arquivos já extraídos por alguém.
    """
    if raiz.is_dir():
        caminho = raiz / relativo
        if not caminho.exists():
            raise DatasetError(
                f"corpus BEIR incompleto: {caminho} nao existe",
                context=ErrorContext(phase=Phase.DATASET_LOAD, details={"raiz": str(raiz)}),
            )
        with caminho.open(encoding="utf-8") as fh:
            yield fh
        return

    with zipfile.ZipFile(raiz) as z:
        # O zip do BEIR traz tudo sob um diretório com o nome do dataset (`scifact/corpus.jsonl`).
        # Casar pelo SUFIXO em vez de montar o prefixo evita depender de o diretório se chamar como
        # o arquivo — o que é verdade no SciFact e não é contrato.
        nomes = [n for n in z.namelist() if n.endswith(relativo)]
        if not nomes:
            raise DatasetError(
                f"corpus BEIR incompleto: {relativo} nao esta em {raiz.name}",
                context=ErrorContext(phase=Phase.DATASET_LOAD, details={"raiz": str(raiz)}),
            )
        with z.open(sorted(nomes, key=len)[0]) as bruto:
            yield io.TextIOWrapper(bruto, encoding="utf-8")


def load_beir(raiz: Path, *, split: str = "test") -> tuple[list[Document], QuerySet]:
    """Lê um corpus BEIR — diretório extraído OU o `.zip` publicado.

    Devolve apenas as consultas QUE TÊM julgamento no split — uma consulta sem qrel não tem verdade
    contra a qual pontuar, e incluí-la faria o nDCG médio cair por uma razão que não é qualidade de
    busca. O SciFact publica 1109 consultas e julga 300 no split `test`.
    """
    documentos: list[Document] = []
    id_doc: dict[str, int] = {}
    with _abrir(raiz, "corpus.jsonl") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            registro = json.loads(linha)
            interno = len(documentos)
            id_doc[str(registro["_id"])] = interno
            titulo = (registro.get("title") or "").strip()
            texto = (registro.get("text") or "").strip()
            documentos.append(
                Document(
                    id=interno,
                    # Título e corpo concatenados: é o que a literatura do BEIR usa, e separá-los
                    # mediria um índice que ninguém constrói.
                    text=f"{titulo} {texto}".strip(),
                    vector=np.zeros(_SEM_VETOR_DIM, dtype=np.float32),
                )
            )

    texto_consulta: dict[str, str] = {}
    with _abrir(raiz, "queries.jsonl") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            registro = json.loads(linha)
            texto_consulta[str(registro["_id"])] = str(registro["text"])

    julgamentos: dict[str, dict[int, float]] = {}
    with _abrir(raiz, f"qrels/{split}.tsv") as fh:
        cabecalho = fh.readline()
        if "query-id" not in cabecalho:
            fh.seek(0)  # arquivo sem cabecalho
        for linha in fh:
            partes = linha.rstrip("\n").split("\t")
            if len(partes) < 3:
                continue
            qid, did, nota = partes[0], partes[1], partes[2]
            achado = id_doc.get(did)
            if achado is None:
                # Um qrel que aponta para documento fora do corpus e um defeito do dataset, nao um
                # zero. Ignorar em silencio inflaria o denominador do recall sem dizer por que.
                raise DatasetError(
                    f"qrel aponta para documento ausente do corpus: {did!r}",
                    context=ErrorContext(phase=Phase.DATASET_LOAD, details={"query": qid}),
                )
            julgamentos.setdefault(qid, {})[achado] = float(nota)

    qids = sorted(julgamentos, key=lambda q: (len(q), q))
    ausentes = [q for q in qids if q not in texto_consulta]
    if ausentes:
        raise DatasetError(
            f"{len(ausentes)} consulta(s) julgada(s) sem texto: {ausentes[:3]}",
            context=ErrorContext(phase=Phase.DATASET_LOAD, details={"split": split}),
        )

    consultas = QuerySet(
        texts=tuple(texto_consulta[q] for q in qids),
        vectors=np.zeros((len(qids), _SEM_VETOR_DIM), dtype=np.float32),
        relevance=tuple(julgamentos[q] for q in qids),
    )
    return documentos, consultas
