"""Esquema analítico multi-tabela, no shape que o concorrente publicou (B-065).

POR QUE ESTE MÓDULO EXISTE. O contrato analítico anterior era de **uma** tabela: `AnalyticalTable`
carrega `name`, `columns` e `path`, e as quatro queries de `bench/analytical.py` são agregação sobre
tabela única. **Nenhuma junção era expressável.**

A avaliação independente do AlloyDB publicou Q1/Q5/Q6/Q18 do TPC-H. Sem esquema multi-tabela, os
números dela não têm onde ser respondidos com o mesmo shape — e responder com shape nosso mede outra
coisa e chama de comparação.

A **Q18 é a que prova o redesenho**: `customer` ⋈ `orders` ⋈ `lineitem`. Q1 e Q6 são de `lineitem`
só e passariam no contrato antigo.

O GERADOR NÃO É O `dbgen`. É um gerador semeado com a MESMA FORMA de esquema e as mesmas colunas que
as três queries tocam — o suficiente para que a comparação seja do shape certo, e explicitamente
insuficiente para reivindicar "resultado TPC-H", que é marca registrada e exige o kit oficial. O
[[B-058]] registra a decisão de licença do dbgen como item próprio, anterior a qualquer corrida.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from theodb_bench.adapters.base import AnalyticalQuery, AnalyticalTable
from theodb_bench.errors import ConfigError, ErrorContext, Phase

Rows = list[tuple[Any, ...]]
Dataset = dict[str, Rows]


@dataclass(frozen=True)
class ForeignKey:
    """Uma aresta do esquema: sem ela, junção não é expressável — só adivinhável."""

    table: str
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class AnalyticalSchema:
    """Várias tabelas e as chaves entre elas. O sucessor de `AnalyticalTable` para junção."""

    tables: tuple[AnalyticalTable, ...]
    keys: tuple[ForeignKey, ...]

    def table(self, name: str) -> AnalyticalTable:
        for t in self.tables:
            if t.name == name:
                return t
        raise ConfigError(
            f"a tabela {name!r} não está no esquema", context=ErrorContext(phase=Phase.OFFLINE)
        )


#: Colunas de cada tabela, na ordem em que o gerador as emite. Subconjunto do TPC-H: exatamente o
#: que Q1, Q6 e Q18 tocam. Gerar colunas que nenhuma query lê inflaria a carga e o tempo de toda
#: corrida sem mudar um número.
_CUSTOMER: Final[tuple[str, ...]] = ("c_custkey", "c_name")
_ORDERS: Final[tuple[str, ...]] = ("o_orderkey", "o_custkey", "o_orderdate", "o_totalprice")
_LINEITEM: Final[tuple[str, ...]] = (
    "l_orderkey",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
    "l_returnflag",
    "l_linestatus",
    "l_shipdate",
)

#: Tipos SQL, na mesma ordem das colunas. `l_shipdate` e `o_orderdate` sao INTEIROS no formato
#: YYYYMMDD, e nao `date`: o filtro das queries e uma comparacao numerica, e converter para `date`
#: obrigaria cada motor a concordar sobre parsing de data antes de concordar sobre desempenho.
_CUSTOMER_TYPES: Final[tuple[str, ...]] = ("integer", "text")
_ORDERS_TYPES: Final[tuple[str, ...]] = ("integer", "integer", "integer", "double precision")
_LINEITEM_TYPES: Final[tuple[str, ...]] = (
    "integer",
    "integer",
    "double precision",
    "double precision",
    "double precision",
    "text",
    "text",
    "integer",
)

TPCH_QUERIES: Final[tuple[AnalyticalQuery, ...]] = (
    AnalyticalQuery(
        id="q1",
        description=(
            "Pricing summary: agrega lineitem por returnflag/linestatus com filtro de data."
        ),
    ),
    AnalyticalQuery(
        id="q6",
        description="Forecasting revenue change: soma filtrada sobre lineitem, sem junção.",
    ),
    AnalyticalQuery(
        id="q18",
        description="Large volume customer: customer x orders x lineitem, três tabelas.",
    ),
)

#: Queries do TPC-H que o concorrente publicou e que NÃO entram, com a razão.
#:
#: O silêncio é que não serve: uma query ausente sem explicação se lê como esquecimento, e a próxima
#: pessoa refaz a análise para chegar à mesma conclusão.
OUT_OF_SCOPE: Final[dict[str, str]] = {
    "q5": (
        "A Q5 junta SEIS tabelas (customer, orders, lineitem, supplier, nation, region) e as duas "
        "últimas são dimensões que nenhuma outra query registrada toca. Gerá-las e mantê-las "
        "consistentes multiplica a superfície do gerador e do oráculo para responder UM número, "
        "enquanto Q1, Q6 e Q18 já cobrem os três shapes que importam: agregação com filtro, "
        "varredura filtrada sem junção, e junção de três tabelas. Entra quando houver uma pergunta "
        "que só ela responda — e a decisão está aqui para que essa reabertura seja deliberada "
        "e não um redescobrimento."
    ),
}


def tpch_schema(*, prefix: str = "", path: str = "row") -> AnalyticalSchema:
    """O esquema e as chaves que a Q18 percorre.

    UM ESPAÇO DE NOMES SÓ, e isso foi um defeito de desenho que o primeiro teste pegou: a versão
    anterior nomeava as tabelas com um prefixo físico (`tpch_orders`) e as chaves com o nome lógico
    (`orders`). Uma junção construída a partir do esquema não resolveria nenhuma das duas pontas —
    e o dado gerado, que é indexado pelo nome lógico, não casaria com nenhuma delas.

    O `prefix` continua disponível para quem precise isolar as tabelas num banco compartilhado, e
    ele se aplica a AMBOS os lados: as chaves são derivadas dos mesmos nomes.

    O `path` decide o access method de TODAS as três tabelas, e não de uma. Uma junção em que
    `lineitem` é colunar e `orders` é heap não mede nem um caminho nem o outro — mede uma terceira
    coisa e a rotula com o nome de um dos dois. É o critério aberto do [[B-058]]: *TPC-H nos mesmos
    moldes, `theodb_columnar` contra heap no MESMO binário*. O default é `row` porque heap é o que
    o PostgreSQL usa sem que ninguém peça, e porque não mudar o comportamento de quem já chamava
    isto é o mínimo.
    """
    customer, orders, lineitem = f"{prefix}customer", f"{prefix}orders", f"{prefix}lineitem"
    return AnalyticalSchema(
        tables=(
            AnalyticalTable(
                name=customer, columns=_CUSTOMER, column_types=_CUSTOMER_TYPES, path=path
            ),
            AnalyticalTable(
                name=orders, columns=_ORDERS, column_types=_ORDERS_TYPES, path=path
            ),
            AnalyticalTable(
                name=lineitem, columns=_LINEITEM, column_types=_LINEITEM_TYPES, path=path
            ),
        ),
        keys=(
            ForeignKey(orders, "o_custkey", customer, "c_custkey"),
            ForeignKey(lineitem, "l_orderkey", orders, "o_orderkey"),
        ),
    )


#: Linhas por unidade de fator de escala, na proporção do TPC-H (150k clientes, 1,5M pedidos, ~6M
#: itens a SF1). Mantida para que o fator de escala signifique o mesmo que significa lá.
_CUSTOMERS_PER_SF: Final[int] = 150_000
_ORDERS_PER_CUSTOMER: Final[int] = 10
_LINEITEMS_PER_ORDER: Final[int] = 4


def generate_tpch(*, scale_factor: float, seed: int) -> Dataset:
    """Dados semeados e reprodutíveis, na forma do esquema.

    SEMEADO importa mais do que parece: sem isso, comparar duas corridas mede a diferença entre os
    DADOS e chama de diferença entre os sistemas. Duas chamadas com a mesma semente produzem bytes
    idênticos, e é isso que o teste trava.
    """
    if scale_factor <= 0:
        raise ConfigError(
            f"o fator de escala precisa ser positivo, recebeu {scale_factor}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    rng = np.random.default_rng(seed)
    n_cust = max(1, int(_CUSTOMERS_PER_SF * scale_factor))
    n_ord = n_cust * _ORDERS_PER_CUSTOMER
    n_item = n_ord * _LINEITEMS_PER_ORDER

    clientes: Rows = [(i, f"Customer#{i:09d}") for i in range(1, n_cust + 1)]

    donos = rng.integers(1, n_cust + 1, size=n_ord)
    datas_ped = rng.integers(19920101, 19981231, size=n_ord)
    pedidos: Rows = [
        (i + 1, int(donos[i]), int(datas_ped[i]), 0.0)  # totalprice preenchido abaixo
        for i in range(n_ord)
    ]

    de_pedido = rng.integers(1, n_ord + 1, size=n_item)
    quantidades = rng.integers(1, 51, size=n_item)
    precos = rng.uniform(900.0, 105_000.0, size=n_item)
    descontos = rng.uniform(0.0, 0.10, size=n_item)
    impostos = rng.uniform(0.0, 0.08, size=n_item)
    flags = rng.choice(np.array(["A", "N", "R"]), size=n_item)
    status = rng.choice(np.array(["F", "O"]), size=n_item)
    datas_env = rng.integers(19920101, 19981231, size=n_item)

    itens: Rows = [
        (
            int(de_pedido[i]),
            int(quantidades[i]),
            round(float(precos[i]), 2),
            round(float(descontos[i]), 4),
            round(float(impostos[i]), 4),
            str(flags[i]),
            str(status[i]),
            int(datas_env[i]),
        )
        for i in range(n_item)
    ]

    # `o_totalprice` derivado dos itens, e não sorteado: um total que não bate com as linhas faria a
    # Q18 medir uma inconsistência do gerador em vez do motor.
    totais: dict[int, float] = {}
    for item in itens:
        totais[item[0]] = totais.get(item[0], 0.0) + item[2] * (1 - item[3])
    pedidos = [(p[0], p[1], p[2], round(totais.get(p[0], 0.0), 2)) for p in pedidos]

    return {"customer": clientes, "orders": pedidos, "lineitem": itens}


#: Limiar da Q18: pedidos cuja soma de quantidades passa disto. O TPC-H usa 300 a SF1; aqui é menor
#: porque o gerador emite menos itens por pedido, e um limiar que nenhuma linha atinge devolve um
#: oráculo vazio — que não detecta nada.
_Q18_QUANTITY_THRESHOLD: Final[int] = 100


def expected_tpch_answer(data: Dataset, query_id: str) -> tuple[tuple[Any, ...], ...]:
    """A resposta correta, calculada AQUI a partir dos mesmos dados.

    É o oráculo, e a junção da Q18 é calculada em Python sem tocar em nenhum motor. Esse é o ponto:
    se os três caminhos medidos concordassem na mesma resposta errada, compará-los entre si não
    acharia nada.
    """
    itens = data["lineitem"]

    if query_id == "q1":
        grupos: dict[tuple[str, str], list[float]] = {}
        for _orderkey, qtd, preco, desc, imposto, flag, stat, envio in itens:
            if envio > 19980901:
                continue
            chave = (flag, stat)
            acc = grupos.setdefault(chave, [0.0, 0.0, 0.0, 0.0])
            acc[0] += qtd
            acc[1] += preco
            acc[2] += preco * (1 - desc)
            acc[3] += preco * (1 - desc) * (1 + imposto)
        return tuple(
            (f, s, round(a[0], 2), round(a[1], 2), round(a[2], 2), round(a[3], 2))
            for (f, s), a in sorted(grupos.items())
        )

    if query_id == "q6":
        receita = sum(
            preco * desc
            for _, qtd, preco, desc, _, _, _, envio in itens
            if 19940101 <= envio < 19950101 and 0.05 <= desc <= 0.07 and qtd < 24
        )
        return ((round(receita, 2),),)

    if query_id == "q18":
        # customer |x| orders |x| lineitem — a junção que o contrato anterior não expressava.
        qtd_por_pedido: dict[int, int] = {}
        for l_orderkey, qtd, *_ in itens:
            qtd_por_pedido[l_orderkey] = qtd_por_pedido.get(l_orderkey, 0) + qtd
        grandes = {k for k, v in qtd_por_pedido.items() if v > _Q18_QUANTITY_THRESHOLD}
        nomes = {c[0]: c[1] for c in data["customer"]}
        saida = [
            (
                nomes[o_custkey],
                o_custkey,
                o_orderkey,
                o_orderdate,
                round(o_total, 2),
                qtd_por_pedido[o_orderkey],
            )
            for o_orderkey, o_custkey, o_orderdate, o_total in data["orders"]
            if o_orderkey in grandes
        ]
        return tuple(sorted(saida, key=lambda r: (-r[4], r[3]))[:100])

    if query_id in OUT_OF_SCOPE:
        raise ConfigError(
            f"{query_id} está fora de escopo: {OUT_OF_SCOPE[query_id]}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    raise ConfigError(
        f"não há oráculo para a query {query_id!r}", context=ErrorContext(phase=Phase.OFFLINE)
    )


def _safe_identifier(name: str) -> str:
    """CITA um identificador, recusando o que não for um. Mesma regra e mesma forma do adapter.

    O `return` CITADO é a metade que faltava no meu primeiro rascunho, e um teste a encontrou:
    `.isalnum()` é Unicode-aware, então `café` passa na validação — e passa CERTO, porque
    `"café"` é identificador legal no PostgreSQL. Errado era devolvê-lo sem aspas para
    interpolação. Validar sem citar aceita o que não deveria e quebra o que deveria funcionar.
    """
    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ConfigError(
            f"identificador SQL inválido: {name!r}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    return f'"{name}"'


def tpch_sql(schema: AnalyticalSchema, query_id: str) -> str:
    """O SQL de uma query registrada, construído A PARTIR DO ESQUEMA.

    Construir do esquema e não de literais é o que impede a divergência: mudar o prefixo das tabelas
    move os dois lados juntos. É a mesma razão pela qual `_query_parameters` do adapter Postgres
    passou a devolver SQL e parâmetros juntos ([[B-063]]) — forma e referência que vivem em lugares
    diferentes acabam divergindo.
    """
    if query_id in OUT_OF_SCOPE:
        raise ConfigError(
            f"{query_id} está fora de escopo: {OUT_OF_SCOPE[query_id]}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    # A ORDEM das tabelas no esquema é o contrato: customer, orders, lineitem. Desempacotar aqui
    # em vez de procurar por nome mantém o SQL independente do prefixo — que é o ponto de gerá-lo do
    # esquema.
    #
    # E os nomes passam pelo validador ANTES de entrar no texto. Eles vêm de uma definição de
    # benchmark e não de entrada de usuário, mas — como o `_identifier` do adapter Postgres diz —
    # uma definição de benchmark ainda é dado, e dado não escreve SQL. O `S608` do ruff está ligado
    # de propósito neste projeto; silenciá-lo por argumento em vez de por validação seria trocar uma
    # garantia por uma opinião.
    cliente, pedido, item = (_safe_identifier(t.name) for t in schema.tables)

    if query_id == "q1":
        return (
            f"SELECT l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice), "
            f"sum(l_extendedprice * (1 - l_discount)), "
            f"sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) "
            f"FROM {item} WHERE l_shipdate <= 19980901 "
            f"GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus"
        )
    if query_id == "q6":
        return (
            f"SELECT sum(l_extendedprice * l_discount) FROM {item} "
            f"WHERE l_shipdate >= 19940101 AND l_shipdate < 19950101 "
            f"AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"
        )
    if query_id == "q18":
        return (
            f"SELECT c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice, "
            f"sum(l.l_quantity) "
            f"FROM {cliente} c "
            f"JOIN {pedido} o ON o.o_custkey = c.c_custkey "
            f"JOIN {item} l ON l.l_orderkey = o.o_orderkey "
            f"GROUP BY c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice "
            f"HAVING sum(l.l_quantity) > {_Q18_QUANTITY_THRESHOLD} "
            f"ORDER BY o.o_totalprice DESC, o.o_orderdate LIMIT 100"
        )
    raise ConfigError(
        f"não há SQL para a query {query_id!r}", context=ErrorContext(phase=Phase.OFFLINE)
    )


@dataclass(frozen=True)
class TpchMeasurement:
    """O que uma query registrada produziu, e se ela produziu a coisa certa."""

    query_id: str
    seconds: float
    matches_oracle: bool
    rows_returned: int


def run_tpch_suite(
    engine: Any, *, scale_factor: float, seed: int, prefix: str = "", path: str = "row"
) -> dict[str, TpchMeasurement]:
    """Carrega o esquema e roda as queries registradas, conferindo cada resposta contra o oráculo.

    `engine` é qualquer coisa com `load_analytical(table, rows)` e `execute_analytical_sql(sql)` —
    um protocolo estrutural em vez de um tipo nominal, porque o que a suíte precisa do adapter são
    duas operações e não a superfície inteira dele (ISP).

    O ORÁCULO É CONSULTADO SEMPRE, e não só quando alguém pede. Uma query rápida e errada não é uma
    query rápida, e comparar os motores entre si não acharia um erro que todos cometessem — é a
    mesma disciplina do `AnalyticalBenchmark` sobre tabela única, agora sobre junção.
    """
    schema = tpch_schema(prefix=prefix, path=path)
    dados = generate_tpch(scale_factor=scale_factor, seed=seed)

    # As chaves do esquema decidem a ORDEM de carga: uma tabela referenciada entra antes da que a
    # referencia, senão uma FK real recusaria a linha. `keys` deixa de ser metadado decorativo e
    # passa a ter consequência — que é o que o portão de código morto cobrou.
    ordem = _load_order(schema)
    for nome_logico in ordem:
        tabela = schema.table(f"{prefix}{nome_logico}")
        engine.load_analytical(tabela, dados[nome_logico])

    medidas: dict[str, TpchMeasurement] = {}
    for query in TPCH_QUERIES:
        esperado = expected_tpch_answer(dados, query.id)
        inicio = time.perf_counter()
        obtido = tuple(engine.execute_analytical_sql(tpch_sql(schema, query.id)))
        decorrido = time.perf_counter() - inicio
        medidas[query.id] = TpchMeasurement(
            query_id=query.id,
            seconds=decorrido,
            matches_oracle=_answers_agree(obtido, esperado),
            rows_returned=len(obtido),
        )
    return medidas


def _load_order(schema: AnalyticalSchema) -> tuple[str, ...]:
    """Tabelas referenciadas antes das que as referenciam, derivado das chaves do esquema.

    E valida cada aresta antes de usá-la: uma chave que aponta para coluna inexistente descreveria
    uma junção impossível, e o SQL só falharia no servidor — tarde, e com mensagem pior. O campo
    `references_column` existe para isto; sem consumidor ele seria decoração, e o portão de código
    morto cobrou exatamente isso.
    """
    logicos = [t.name for t in schema.tables]
    for chave in schema.keys:
        alvo = schema.table(chave.references_table)
        if chave.references_column not in alvo.columns:
            raise ConfigError(
                f"a chave {chave.table}.{chave.column} aponta para "
                f"{chave.references_table}.{chave.references_column}, que não existe — "
                f"as colunas de {alvo.name} são {', '.join(alvo.columns)}",
                context=ErrorContext(phase=Phase.OFFLINE),
            )
    referentes = [k.table for k in schema.keys]
    grau = {n: sum(1 for r in referentes if r == n) for n in logicos}
    ordenado = sorted(logicos, key=lambda n: grau[n])
    # Os nomes lógicos são os físicos sem o prefixo, e o dado gerado é indexado pelos lógicos.
    comum = _common_prefix(logicos)
    return tuple(n[len(comum) :] for n in ordenado)


def _common_prefix(nomes: Sequence[str]) -> str:
    if not nomes:
        return ""
    for tamanho in range(min(len(n) for n in nomes), 0, -1):
        candidato = nomes[0][:tamanho]
        if all(n.startswith(candidato) for n in nomes) and candidato.endswith("_"):
            return candidato
    return ""


def _answers_agree(
    observado: tuple[Any, ...], esperado: tuple[tuple[Any, ...], ...], tolerance: float = 1e-2
) -> bool:
    """Compara com tolerância em ponto flutuante, e exige a mesma forma."""
    if len(observado) != len(esperado):
        return False
    for linha_obs, linha_esp in zip(observado, esperado, strict=True):
        if len(linha_obs) != len(linha_esp):
            return False
        for a, b in zip(linha_obs, linha_esp, strict=True):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if abs(float(a) - float(b)) > tolerance:
                    return False
            elif a != b:
                return False
    return True
