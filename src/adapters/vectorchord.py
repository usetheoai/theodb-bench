"""VectorChord: PostgreSQL with the `vchordrq` access method — RaBitQ in the database.

Este adapter existe por um motivo nomeado: o [[B-057]] fechou os eixos de memória e build do nosso
`theodb_hnsw` contra o `scann` AH do AlloyDB Omni e contra o `hnsw` do pgvector, e deixou **um** eixo
explicitamente não medido — o **RaBitQ**, que o ADR-0036 registra como o melhor quantizador
permissivo. O VectorChord o implementa dentro do PostgreSQL, que é a forma em que ele nos interessa.

Tudo abaixo foi **medido contra `tensorchord/vchord-postgres:pg17-v0.4.3` rodando**, em 2026-08-24,
nunca lido de documentação:

  * a imagem serve **PostgreSQL 17.4**, enquanto o TheoDB é 18 — um head-to-head cruza uma versão
    maior, e é por isso que `export_config` (herdado) lê a versão DO SERVIDOR;
  * `CREATE EXTENSION vchord CASCADE` traz o `vector` 0.8.0 junto. É por isso que este adapter herda
    `PgvectorAdapter`: o tipo `vector`, os operadores `<->`/`<#>`/`<=>` e o formato de entrada em
    texto são os mesmos;
  * `shared_preload_libraries = vchord.so` na imagem, então as oito GUCs `vchordrq.*` **já estão
    registradas** ao conectar — diferente do Omni, que precisa de `LOAD 'alloydb_scann'`. Por isso
    `library` fica `None`, e não por esquecimento;
  * o AM chama-se **`vchordrq`** e nomeia as opclasses do pgvector (`vector_l2_ops` e irmãs), não
    nomes próprios como o `scann` faz;
  * as opções de build **não são reloptions soltas**: são um bloco TOML dentro de
    `WITH (options = $$ … $$)`. É a única razão de `index_ddl` ser sobrescrito aqui;
  * o botão de busca **tem efeito medido**, não apenas aceito: `vchordrq.probes` de 1 para 64 move
    `Buffers: shared hit` de **15 para 2339** no mesmo plano `Index Scan`. Esse teste é o que o
    [[B-057]] aprendeu com a armadilha do `scann.num_leaves_to_search`, que é aceito e ignorado sem
    o `LOAD` — e mediria o concorrente num default raso, publicando vantagem falsa nossa.

**Licença.** VectorChord é AGPL-3.0. Medir contra ele é legítimo e não distribui nada — a mesma
postura do AlloyDB Omni, que é proprietário. O que a D1 proíbe é **código** copiado para a nossa
distribuição, e nada aqui lê o código deles.
"""

from __future__ import annotations

from typing import Any, ClassVar

from theodb_bench.adapters.base import IndexSpec, VectorTableSpec
from theodb_bench.adapters.postgres import PgvectorAdapter, _identifier
from theodb_bench.errors import AdapterError, ErrorContext, Phase


class VectorChordAdapter(PgvectorAdapter):
    """VectorChord, exercitado pelo seu access method `vchordrq` (RaBitQ)."""

    system_id = "vectorchord"
    extension = "vchord"

    #: As GUCs `vchordrq.*` já vêm registradas: a imagem carrega `vchord.so` em
    #: `shared_preload_libraries` (medido). Nenhum LOAD é necessário — ao contrário do Omni.
    library: ClassVar[str | None] = None

    #: Lidas de `pg_opclass` no servidor. O `vchordrq` reusa os nomes do pgvector; as linhas de
    #: `hnsw`/`ivfflat` vêm do `vector` que o CASCADE instalou, e ficam para permitir um
    #: VectorChord-contra-VectorChord entre RaBitQ e HNSW no mesmo servidor.
    OPCLASSES: ClassVar[dict[str, dict[str, str]]] = {
        "rabitq": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
        "hnsw": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
        "ivfflat": {"l2": "vector_l2_ops", "ip": "vector_ip_ops", "cosine": "vector_cosine_ops"},
    }

    #: O rótulo da família (`rabitq`) não é o nome do AM (`vchordrq`).
    ACCESS_METHODS: ClassVar[dict[str, str]] = {"rabitq": "vchordrq"}

    #: `rq_probes` e não `probes`: o pgvector já usa `probes` para `ivfflat.probes`, e um nome
    #: repetido faria a mesma varredura significar coisas diferentes conforme o motor. Nome mais
    #: específico vence (CLAUDE.md § 5).
    SEARCH_PARAMETERS: ClassVar[frozenset[str]] = frozenset(
        {"rq_probes", "rq_epsilon", "rq_max_scan_tuples"}
    )

    def capabilities(self) -> dict[str, bool]:
        """O que ESTE caminho de código exercita — não o que o VectorChord sabe fazer.

        O `vchordg` (grafo) e o `maxsim` multi-vetor existem no produto e não são alcançados aqui.
        Declarar capacidade que este adapter não exercita faria uma corrida medir outro produto.
        """
        return {
            "vector_exact": True,
            "vector_rabitq": True,
            "vector_hnsw": True,
            "vector_ivfflat": True,
            "vector_filtered": True,
        }

    def _search_guc_mapping(self, parameters: dict[str, Any]) -> dict[str, str]:
        """Os botões de busca do `vchordrq`, pelos nomes que `pg_settings` usa.

        Só os que uma varredura de fato mexe. `lists` é opção de build dentro do TOML, não GUC de
        busca, e não é aceita aqui para que um erro de digitação não passe por parâmetro de busca.
        """
        mapping: dict[str, str] = {}
        for name, value in parameters.items():
            if name == "rq_probes":
                # Quantas listas são varridas. Medido: 1 → 64 move os buffers lidos de 15 para 2339.
                mapping["vchordrq.probes"] = str(int(value))
            elif name == "rq_epsilon":
                mapping["vchordrq.epsilon"] = str(float(value))
            elif name == "rq_max_scan_tuples":
                mapping["vchordrq.max_scan_tuples"] = str(int(value))
        return mapping

    def index_ddl(self, spec: VectorTableSpec, index: IndexSpec) -> tuple[str, str]:
        """As opções do `vchordrq` são um bloco TOML, não reloptions soltas.

        `WITH (lists = 64)` é rejeitado pelo AM; o que ele aceita é
        `WITH (options = $$ [build.internal] \\n lists = [64] $$)`. Qualquer outra família cai no
        caminho herdado, que é o do pgvector.
        """
        if index.kind != "rabitq":
            return super().index_ddl(spec, index)

        opclass = self.opclass(index.kind, spec.metric)
        name = f"{spec.table}_{index.kind}_{spec.metric}_idx"

        lists = index.parameters.get("lists")
        if lists is None:
            raise AdapterError(
                "vchordrq exige `lists` na IndexSpec: sem ele o AM não tem particionamento "
                "declarado e a corrida mediria um default que a suíte não escolheu",
                context=ErrorContext(phase=Phase.INDEX_BUILD, details={"index": index.label()}),
            )
        residual = bool(index.parameters.get("residual_quantization", True))

        toml = (
            f"residual_quantization = {str(residual).lower()}\n"
            "[build.internal]\n"
            f"lists = [{int(lists)}]\n"
        )
        ddl = (
            f"CREATE INDEX {_identifier(name)} ON {_identifier(spec.table)} "
            f"USING vchordrq ({_identifier(spec.embedding_column)} {opclass}) "
            f"WITH (options = $theodb${toml}$theodb$)"
        )
        return name, ddl
