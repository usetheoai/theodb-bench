"""Which capabilities any registered adapter can actually exercise.

The project declares fourteen capability surfaces. Crossing that vocabulary with
what the adapters declare is the only honest answer to "what can this harness
measure?", and doing it by hand is how the answer goes stale: the measurement
that opened B-073 found six of fourteen reachable, and a table typed that day
would still say six after the number moved.

So it is derived from the same registry the runs use. That has a second property
worth more than freshness: a capability nobody declares shows up as an empty row
rather than as an absent one. Absent reads as "not a thing"; empty reads as
"nothing reaches this yet", which is both true and the statement someone needs
before promising a pillar.

The `fake` adapter is excluded. It exists to test the harness, and counting it
would let the matrix report a pillar as reachable when no real system implements
it — the same shape of false green the rest of this project spends its effort
avoiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from theodb_bench.adapters.base import CAPABILITIES
from theodb_bench.registry import ADAPTERS, BENCHMARKS

#: Adapters that exist to exercise the harness rather than a system.
_NOT_A_SYSTEM: Final[frozenset[str]] = frozenset({"fake"})


@dataclass(frozen=True)
class CapabilityRow:
    """One capability and the real adapters that declare it."""

    capability: str
    adapters: tuple[str, ...]
    #: Suítes registradas que EXERCITAM esta capacidade.
    #:
    #: Uma capacidade declarada por um adapter diz que o sistema **sabe** fazer aquilo. Ela
    #: não diz que alguém **mede**. As duas colunas juntas mostram a diferença, que é a
    #: distinção central deste projeto — e que a matriz não mostrava.
    suites: tuple[str, ...] = ()

    @property
    def reachable(self) -> bool:
        return bool(self.adapters)

    @property
    def declared_unmeasured(self) -> bool:
        """Alguém sabe fazer, e ninguém mede."""
        return bool(self.adapters) and not self.suites


#: Como uma suíte registrada revela a capacidade que exercita.
#:
#: Derivado do REGISTRO, não de uma lista à mão: o `kind` do índice e o prefixo do id são
#: dados que a própria suíte declara, e por isso esta tabela não pode desatualizar em silêncio
#: quando alguém registra uma suíte nova. Foi a exigência do [[B-104]], e a razão é que uma
#: tabela escrita à mão sobre cobertura mordeu este projeto sete vezes num único dia.
#:
#: Capacidades que nenhum sinal do registro alcança — `hybrid`, `rerank`, `vectorizer`,
#: `ai_sql` — saem com zero, e **zero é a resposta certa**: nenhuma suíte as exercita.
_KIND_TO_CAPABILITY: Final[dict[str, str]] = {
    "none": "vector_exact",
    "hnsw": "vector_hnsw",
    "ivfflat": "vector_ivfflat",
    "scann": "vector_scann",
}


def suites_by_capability() -> dict[str, tuple[str, ...]]:
    """Quais suítes registradas exercitam cada capacidade, derivado do registro."""
    encontrado: dict[str, list[str]] = {capability: [] for capability in CAPABILITIES}

    def marcar(capability: str, suite: str) -> None:
        if capability in encontrado and suite not in encontrado[capability]:
            encontrado[capability].append(suite)

    for suite, entrada in sorted(BENCHMARKS.items()):
        workload = entrada.workload
        for index in getattr(workload, "indexes", ()) or ():
            capability = _KIND_TO_CAPABILITY.get(index.kind)
            if capability:
                marcar(capability, suite)
            # Um índice com quantizador declarado exercita a quantização, e o `kind`
            # sozinho não diria isso.
            if (index.parameters or {}).get("quantizer"):
                marcar("vector_quantized", suite)
        if getattr(workload, "filter_cardinality", None) is not None:
            marcar("vector_filtered", suite)
        for path in getattr(workload, "paths", ()) or ():
            if path in ("columnar", "parquet"):
                marcar(path, suite)
        if suite.startswith("retrieval/"):
            marcar("lexical", suite)
        if suite.startswith("graph/"):
            marcar("graph", suite)
    return {capability: tuple(nomes) for capability, nomes in encontrado.items()}


def capability_matrix() -> list[CapabilityRow]:
    """Every declared capability, in vocabulary order, with who reaches it."""
    declared: dict[str, list[str]] = {capability: [] for capability in CAPABILITIES}
    for name, entry in sorted(ADAPTERS.items()):
        if name in _NOT_A_SYSTEM:
            continue
        for capability in entry.factory().capabilities():
            if capability in declared:
                declared[capability].append(name)
    medindo = suites_by_capability()
    return [
        CapabilityRow(
            capability=capability,
            adapters=tuple(names),
            suites=medindo.get(capability, ()),
        )
        for capability, names in declared.items()
    ]


def render_capability_matrix() -> str:
    """The matrix as a Markdown table, for the README and for the terminal."""
    matrix = capability_matrix()
    reachable = sum(1 for row in matrix if row.reachable)

    lines = [
        f"**{reachable} of {len(CAPABILITIES)} capabilities are reachable by a real adapter.**",
        "",
        "| capability | adapters que declaram | suítes que medem |",
        "|---|---|---|",
    ]
    for row in matrix:
        adapters = ", ".join(f"`{name}`" for name in row.adapters) if row.adapters else "—"
        medem = str(len(row.suites)) if row.suites else "**0**"
        lines.append(f"| `{row.capability}` | {adapters} | {medem} |")
    lines += [
        "",
        "A dash is not a gap in this table — it is the measured state. `rerank`, "
        "`vectorizer` and `ai_sql` each reach an external model, and without an "
        "endpoint there is nothing to measure; a stub would put a number where an "
        "absence belongs.",
    ]
    orfas = [row.capability for row in matrix if row.declared_unmeasured]
    if orfas:
        lines += [
            "",
            "**Declarada e não medida: "
            + ", ".join(f"`{c}`" for c in orfas)
            + ".** Um adapter declara que o sistema sabe fazer aquilo, e nenhuma suíte "
            "registrada o exercita — as duas colunas dizem coisas diferentes, e a segunda "
            "é a que sustenta um número. Ver B-104.",
        ]
    return "\n".join(lines)
