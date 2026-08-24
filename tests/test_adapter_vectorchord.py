"""O adapter do VectorChord: o que dá para provar sem servidor.

Cada expectativa aqui foi **medida** contra `tensorchord/vchord-postgres:pg17-v0.4.3` rodando, em
2026-08-24, antes de ser escrita:

  * a imagem serve PostgreSQL 17.4, uma versão maior atrás do TheoDB;
  * `CREATE EXTENSION vchord CASCADE` instala `vector` 0.8.0 junto;
  * `shared_preload_libraries = vchord.so`, e `pg_settings` já lista as oito GUCs `vchordrq.*` sem
    nenhum `LOAD` — o oposto do Omni, e por isso `library` é `None` aqui;
  * `pg_opclass` dá `vchordrq -> vector_l2_ops` (e `ip`/`cosine`), os nomes do pgvector;
  * `WITH (lists = 64)` é recusado; o AM aceita `WITH (options = $$ … TOML … $$)`;
  * `vchordrq.probes` de 1 para 64 move `Buffers: shared hit` de 15 para 2339 no mesmo
    `Index Scan` — o botão TEM efeito, que é diferente de ser aceito.

O último item é a lição que o `scann.num_leaves_to_search` cobrou: lá o `SET` é aceito, ecoado de
volta e ignorado sem o `LOAD`, e uma corrida que não conferisse mediria o concorrente num default
raso e publicaria vantagem falsa nossa.
"""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import IndexSpec, VectorTableSpec
from theodb_bench.adapters.vectorchord import VectorChordAdapter
from theodb_bench.errors import AdapterError


def _adapter() -> VectorChordAdapter:
    return VectorChordAdapter.__new__(VectorChordAdapter)


def _spec() -> VectorTableSpec:
    return VectorTableSpec(table="sift", embedding_column="e", dimension=128, metric="l2")


def test_o_am_do_rabitq_e_vchordrq_e_nao_o_rotulo_da_familia():
    """O rótulo que o bundle reporta (`rabitq`) não é o nome que o motor usa (`vchordrq`)."""
    _, ddl = _adapter().index_ddl(_spec(), IndexSpec(kind="rabitq", parameters={"lists": 64}))
    assert "USING vchordrq" in ddl
    assert "vector_l2_ops" in ddl


def test_as_opcoes_de_build_saem_como_bloco_toml():
    """`WITH (lists = 64)` é recusado pelo AM — medido. O que ele aceita é o TOML."""
    _, ddl = _adapter().index_ddl(_spec(), IndexSpec(kind="rabitq", parameters={"lists": 128}))
    assert "WITH (options = $theodb$" in ddl
    assert "[build.internal]" in ddl
    assert "lists = [128]" in ddl
    assert "WITH (lists" not in ddl


def test_sem_lists_o_build_e_RECUSADO_em_vez_de_cair_num_default():
    """Um default que a suíte não escolheu é um número que ninguém declarou."""
    with pytest.raises(AdapterError, match="lists"):
        _adapter().index_ddl(_spec(), IndexSpec(kind="rabitq"))


def test_o_botao_de_busca_nao_se_chama_probes():
    """`probes` já é do `ivfflat` no pgvector; repetir o nome faria a mesma varredura significar
    coisas diferentes conforme o motor."""
    a = _adapter()
    assert a._search_guc_mapping({"rq_probes": 32}) == {"vchordrq.probes": "32"}
    assert a._search_guc_mapping({"probes": 32}) == {}
    assert "probes" not in VectorChordAdapter.SEARCH_PARAMETERS


def test_nao_declara_capacidade_que_este_caminho_nao_exercita():
    """`vchordg` e o multi-vetor `maxsim` existem no produto e não são alcançados por este adapter."""
    caps = _adapter().capabilities()
    assert caps["vector_rabitq"] is True
    assert "vector_maxsim" not in caps
    assert "columnar" not in caps


def test_nenhum_load_e_emitido_porque_a_imagem_pre_carrega_a_biblioteca():
    """Medido: `pg_settings` lista as oito GUCs `vchordrq.*` numa sessão que nunca deu LOAD."""
    assert VectorChordAdapter.library is None
