# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **A prova de que a consulta analítica usou o caminho que ela declara nunca era pedida.**
  `assert_analytical_path` existia no `PostgresAdapter` — provando residência pelo `pg_class.relam`
  **e** que o plano usou o caminho, pelo `ANALYTICAL_PLAN_MARKERS` — e o `AlloyDbOmniAdapter` chegou a
  estendê-la com três fatos separados (engine ligado, store populado, planner preferindo colunar).
  **Zero chamadas em `src/bench/`**: o único chamador no repositório era um `super()` dentro do próprio
  override. É o defeito que o #B-063 documenta sobre o `assert_index_used`, repetido para a outra
  família — e a base sequer declarava o método, então o arnês não tinha como pedi-lo a um adapter
  qualquer.
  .
  O que isso permitia é o que o #B-058 registra ter custado uma corrida inteira a um avaliador
  terceiro: **medir heap sob o rótulo do colunar**, sem erro e sem aviso. Medido no próprio adapter a
  um milhão de linhas, a MESMA tabela colunar leva 1407 ms com a pushdown desligada e 108 ms com ela
  ligada — 13×, decidido por um GUC, com o catálogo reportando colunar nos dois casos. Residência
  prova onde as linhas estão; só o plano prova o que rodou.
  .
  A prova passa a ser pedida **antes de qualquer cronometragem**, e um caminho que não se prova produz
  medida `invalid` com a razão — nunca um número. Quatro testes, um deles afirmando que **nada** foi
  cronometrado quando a prova falha. (#B-058)

### Added
- **A família de retrieval deixou de ser órfã, e o pilar lexical passou a ter corpus com julgamento
  humano.** O `bench/retrieval.py` trazia nDCG@10, Recall@k, MRR e quatro pipelines desde sempre, e
  **nenhum benchmark registrado o alcançava** — ele estava na lista de órfãos do próprio arnês, e o
  único corpus era o semeado, cuja docstring diz que exercita a pipeline sem ser alegação de
  qualidade. A consequência foi concreta: todo número lexical publicado saiu de script ad-hoc, e o
  `m186` atribuiu ao PRODUTO um limite que era do script.
  .
  Entram: o `RetrievalWorkload` satisfazendo o protocolo `Workload` (os cinco membros faltavam), o
  `points()` no benchmark, o carregador `bench/beir.py`, o manifesto `beir-scifact` verificado por
  sha256, e o benchmark registrado `retrieval/scifact/lexical`. O `bench.retrieval` **saiu do
  baseline de órfãos** — que existe para encolher. (#B-093)
- **Medido de ponta a ponta: nDCG@10 de 0,6864 no SciFact** (recall@10 0,8227, QPS 213,5, CV 2,1%),
  contra os **0,6269** que o `m186` publicou somando scores por termo do lado de fora. **+9,5%
  relativo** — o número antigo era um piso, como o [[B-014]] havia previsto ao medir que
  `bm25_search` sempre aceitou consulta multi-termo. É o primeiro número lexical deste projeto
  produzido **dentro do arnês**. (#B-093)

### Added
- **`vector/sift1m/frontier`** — varredura de `ef_search` em {40, 64, 128, 256} num único build, para
  ler **dois motores a recall casado** em vez de a `ef` casado. Comparar QPS no mesmo `ef` compara a
  coisa errada: `ef` não é a mesma unidade em dois grafos diferentes — no mesmo 64 o [[B-046]] mede
  recall 0,9600 nosso contra 0,9835 do pgvector, e um "déficit" lido nesse par compara quem buscou
  menos com quem buscou mais. Os quatro pontos saem do mesmo índice (`ef_search` é GUC de sessão), e
  o artefato já registra `build_seconds` e `index_size_bytes`, o que responde o [[B-042]] sem uma
  segunda corrida. (#B-046, #B-042)

### Fixed
- **O arnês não conseguia buscar o dataset que ele mesmo declara.** Medido num host limpo: a origem do
  `sift-128-euclidean` responde **403 Forbidden** ao `User-Agent` default do `urllib`
  (`Python-urllib/3.12`) e **200** ao do `curl` — o CDN filtra por agente. O defeito só não aparecia
  porque toda corrida anterior encontrou o arquivo já em disco, que é o modo de falha que só um host
  limpo revela. O agente passa a **identificar o cliente** em vez de fingir ser navegador: um arnês cuja
  premissa é medir honestamente não começa mentindo na primeira requisição, e há teste fixando isso.
- **A causa de um erro deixou de sumir na mensagem legível.** Uma corrida abortou com
  `could not connect to theodb [phase=bootstrap system=theodb]` e nada mais; a causa real —
  `FATAL: role "root" does not exist` — estava anexada em `cause` e ia para o `as_dict()`, mas o
  `system.log` do bundle formata `{exc}`, ou seja, o `__str__`, que a descartava. **Diagnosticar custou
  duas corridas de benchmark** com a resposta parada no log do servidor desde o primeiro segundo. Um
  lugar consertado, toda renderização humana corrigida.

### Added
- **`vector/sift1m/ef-default`** — SIFT1M contra o `theodb_hnsw` nos **dois defaults de `ef_search` em
  disputa**: 40 (do pgvector) e 64 (nosso). Existe porque o [[B-018]] mediu que o planner larga o índice
  numa junção com filtro seletivo em `ef=64`, e que o pgvector no MESMO 64 produz plano e custos
  idênticos aos nossos — logo a diferença é a escolha do default, não a implementação. Baixá-lo é uma
  linha e **troca recall por plano**; este benchmark é o que torna essa troca medida em vez de suposta.
  A varredura {40, 64} responde "o que o default custa"; a {64, 256} das outras suítes responde "onde
  fica a fronteira", que é outra pergunta. (#B-018)

## [0.4.0] - 2026-08-21

### Added
- **As diferenças de VELOCIDADE que publicamos passaram a ter teste — e o pareado não servia para
  elas.** O [[B-045]] deu rigor à *qualidade* (paridade lexical do `b047`, p=0,477 sobre 6.980
  consultas) enquanto as duas maiores diferenças do projeto seguiam sem nenhum: Elasticsearch a
  **4,3x** o nosso QPS e pgvector a **+16,3%**. O pareado precisa de valor por consulta e QPS é uma
  taxa agregada — pareá-lo inventaria correlação e estreitaria o intervalo sem razão. Entram
  `compare_throughput` (Welch para amostras independentes + bootstrap sobre a **razão**, que é o que
  a frase publica), `precision_for_n`, o veredito em `compare.py` nomeando o teste, e o comando
  `theodb-bench throughput`. O `_welch_p_value` e o `_t_critical` são próprios — para não trazer
  SciPy por duas funções — e **validados contra referência externa**: p bate com
  `scipy.stats.ttest_ind(equal_var=False)` a ~1e-15, t crítico bate com as tabelas na quarta casa.
  Aplicado retroativamente: o **b035 sobrevive** com IC de ±8,2% em vez da precisão que "+16,3%"
  sugere, e o **b047 não é testável** — tem uma corrida por configuração. (#B-049)

- **O contrato analítico deixou de ser de uma tabela só.** A avaliação independente do AlloyDB
  publicou Q1/Q5/Q6/Q18 do TPC-H, e a **Q18 junta três tabelas** — nenhuma junção era expressável, e
  responder com shape nosso mede outra coisa e chama de comparação. Entra o esquema multi-tabela com
  chaves, um gerador semeado e reprodutível, o **oráculo da junção calculado em Python** (se os três
  caminhos concordassem no mesmo erro, compará-los entre si não acharia nada), o SQL construído a
  partir do esquema, e o comando `theodb-bench tpch`. Verificado contra um PostgreSQL real: as três
  queries batem com o oráculo. A **Q5 fica fora de escopo com a razão escrita** — seis junções, duas
  dimensões que nenhuma outra query toca. (#B-065)
- **`execute_analytical_sql` e tipos de coluna por tabela.** `execute_analytical` recebe UMA tabela,
  então uma junção de três não passa por ela; e o carregador criava um esquema fixo e depois copiava
  em colunas de outro nome, o que falha na primeira linha. (#B-065)

- **A contenção escrita x scan passou a ser medível.** Não havia carga mista: existia motor
  concorrente (`run_load`) e o pilar vetorial o usava, mas nada rodava um escritor ao mesmo tempo que
  leitores — que é onde a avaliação independente do AlloyDB mediu uma **inversão** (ligar o colunar
  **piorou** a contenção a SF100: 29% contra 16% do row store). O arnês mede cada lado sozinho e os
  dois juntos **na mesma sessão**, reporta a degradação como **razão** contra a própria linha de
  base, e **recusa** quando um lado não completou nenhuma operação — `null` sobre zero sucessos se lê
  como "sem contenção" e significa "nada rodou". O regime (memória ou além do cache) é **declarado**,
  nunca inferido. (#B-066)
- **`append_analytical_row` no adapter Postgres** — a escrita de primeiro plano que a contenção
  precisa. SQL e parâmetros vêm **juntos** do mesmo método: separá-los foi o que produziu o defeito
  do #B-063. (#B-066)
## [0.3.0] - 2026-08-20

### Fixed
- **A versão que o pacote reporta deixou de poder divergir da tag da release.** `pyproject.toml` e
  `src/__init__.py` declaravam **dois** literais `0.1.0.dev0`, e nenhum foi tocado nos cortes da
  `v0.1.0` nem da `v0.2.0` — o `release.yml` deriva tudo da tag e do CHANGELOG, e nunca lia nenhum
  dos dois. Não é cosmético: `environment.py` grava `theodb-bench {__version__}` na captura de
  ambiente, que entra no bundle, e um bundle existe para ser reproduzível por terceiros. Agora há
  uma fonte só (`importlib.metadata`) e um portão que reprova o corte quando pacote e tag divergem.
  **Nenhum bundle publicado registrou a versão errada** — medido: nenhum artefato fora das fixtures
  de teste carrega o campo. (#B-087)

### Added
- **`PUBLICATION.md` passou a dizer que medição tomada fora do arnês não é publicável** — e por quê,
  com o caso medido: três medições corretas de 2026-08-17 saíram de scripts avulsos e nenhuma era
  reproduzível por terceiros. A segunda razão, que se subestima: um script que contorna o arnês também
  contorna os **defeitos** dele, e três bugs reais daquele mesmo dia só apareceram porque o trabalho
  foi forçado de volta pelo `theodb-bench run`. (#B-069)

## [0.2.0] - 2026-08-20

### Fixed
- **O arnês verifica o caminho de acesso em vez de só nomeá-lo.** `assert_index_used` existia,
  estava escrito com a disciplina certa, era citado por outro item como exemplar — e **não tinha
  chamador nenhum**, além de levantar `ProgrammingError` se alguém o chamasse (dois placeholders,
  um parâmetro). Uma corrida podia reportar `Seq Scan` sob o nome de um índice e nada notaria.
  Agora todo índice construído tem de aparecer no plano, verificado uma vez na janela não
  cronometrada. Nenhum número publicado é retratado: no tamanho da suíte registrada o planner
  escolhe o índice nos três motores. (#B-063)

### Added
- **Um módulo de `analysis/` ou `bench/` não pode mais ficar sem chamador em silêncio.** Seis
  estavam implementados e desconectados, incluindo o núcleo estatístico — código testado por
  unidade que nenhuma corrida jamais executava. Três foram ligados; os três que restam estão
  NOMEADOS num baseline que o teste obriga a encolher: ele falha tanto para um órfão novo quanto
  para um do baseline que já ganhou importador. (#B-071)
- **Detector de código morto na esteira.** O projeto não tinha nenhum, e é por isso que um método
  sem chamador sobreviveu a toda a vida do arquivo. Roda em confiança 60 e não 80 por medição — em
  80 o portão passaria limpo sobre a própria classe de defeito que o motivou. (#B-063)

## [0.1.0] - 2026-08-20

### Added
- Three shapes of vector query beyond "top-10, one vector at a time", which is what eleven
  of the twelve registered suites asked: `vector/sift1m/k-sweep`, `vector/sift1m/filtered`
  and `vector/sift1m/batch` (#B-073)
- **Filtered retrieval is measurable end to end.** A corpus can be partitioned into tenants
  and every query filtered to one, with recall scored against an oracle that filters too —
  the only thing that tells "fast and right" apart from a graph index whose edges cross the
  filter and answers fast and wrong (#B-073)
- **A batch of probes is one round trip**, reported as one operation, so throughput is
  batches per second. A system with no batch path is recorded as `unsupported` rather than
  answered as N single queries, which would make every system look like it batches (#B-073)
- `k` can be swept inside a run. The oracle is computed once at the largest k and sliced,
  and the label carries the k, so two points measured at different k cannot be read as one
  (#B-073)
- A workload is refused at pre-flight when `k` exceeds the corpus, when a filter leaves
  fewer rows per tenant than `k`, or when a batch is larger than the declared query set —
  each of which would score the workload's own arithmetic as the system missing neighbours
  (#B-073)
- A run that fails as unreachable now records whether the system *restarted* — the bundle
  says "went down and came back" or "stayed up while the connection broke", instead of
  leaving the reader to open `docker logs` and `dmesg` (#B-076)
- `theodb-bench capabilities` generates the capability x adapter matrix from the registry,
  and the README's table is now that output rather than a hand-typed one (#B-073)
- `run --build-timeout SECONDS` raises the budget for bulk phases (index build, dataset
  load), which is what the harness already tells the operator to do when one aborts (#B-075)
- Reference scale at 20 000 000 real SIFT descriptors: `vector/bigann20m/hnsw` and
  `vector/bigann20m/load`, with the `bigann-20m-euclidean` dataset manifest (#B-073)
- BIGANN `bvecs` corpora can be measured without being held in memory: a corpus is read
  in row ranges, and `--dataset` accepts one (#B-073)
- Ground truth over a corpus that does not fit in memory, pinned equivalent to the
  resident oracle including its tie-break by ascending id (#B-073)
- **A corpus can be loaded without ever being resident** (billion-scale). `load_dataset` takes an array, so
  the whole corpus had to fit in memory: 1e9 x 128 float32 is 512 GB of RAM. `load_dataset_streaming` reads a
  `CorpusSource` chunk by chunk, so the ceiling becomes disk rather than memory. Binary COPY only — the text
  path would reinstate the per-value Python encoding that is 96% of a load, and at this scale that cost *is*
  the load. Row ids travel with the chunk rather than coming from a counter the caller keeps: a resumed load
  would otherwise renumber every row after the break, and those ids are what a dataset's published neighbour
  lists point at.
- **Ground truth can be scored without reading the corpus** (billion-scale). Brute force is a Q x N product —
  1e13 distance computations for a billion rows and ten thousand queries. `neighbour_vectors` fetches only the
  k x Q vectors the published neighbour ids name, reads each distinct row once because queries share
  neighbours, coalesces contiguous runs into single reads, and reports how many rows it actually read so a run
  can state that instead of implying it read everything. Published *distances* are still never used; they carry
  someone else's precision and metric convention.
- A neighbour id outside the corpus is **refused** (billion-scale). It happens when a published dataset is
  subsampled without remapping its neighbour lists, and dropping such ids quietly would *raise* recall by
  removing exactly the neighbours a system failed to find.

### Changed
- pgvector needing `ORDER BY` to repeat the distance expression (rather than name its alias)
  is now a declared property of the adapter instead of a second copy of `_query_sql` and
  `execute`. The two copies were how a filter added to one would silently miss the other
  (#B-073)
- A vector benchmark takes its corpus through one abstraction with two implementations
  (resident array, streamed source) instead of assuming an array at both call sites;
  `head2head` now loads through it too, so it works at streamed scale (#B-073)
- The metric arithmetic behind ground-truth distances is one implementation shared by
  both corpus shapes, rather than one per shape (#B-073)
- What a billion vectors costs is now written down rather than assumed: **512 GB** of raw float32, **520 GB** in
  a `vector(128)` table, roughly **780 GB** with an HNSW index, and **4.7 hours** of load at the binary-COPY
  rate this harness reaches. The host this was measured on had **284 GB** free, so the capability is present
  and the run needs a larger machine. A benchmark whose scale claims outrun its measurements is worse than one
  whose limits are stated.

- **The graph, hybrid and quantized pillars are reachable** (B-073), taking the count from six of fourteen
  capabilities to **eleven**. Verified against a real server: `theodb.graph_build` folds a CSR over 1 334 edges
  in 0.04 s (98 KB) and `graph_expand` grows correctly with hops (3 → 6 → 10 vertices); `ai.hybrid_search_rrf`
  fuses both legs and ranks the document that matches lexically *and* is the query vector first, at 0.0328
  against a tight 0.0156–0.0161 for the rest.
- Traversing a graph whose CSR was never folded is **refused** (B-073). `graph_expand` answers with an empty
  set, and an empty neighbourhood is a legitimate answer for an isolated vertex — the two are
  indistinguishable after the fact, which is the same shape as the BM25 case.
- A traversal reports `edges_visited` from `graph_expand_card`, asked of the engine rather than inferred
  (B-073). A traversal returning few vertices after walking many edges is expensive, and the answer size alone
  would hide that.
- **The hybrid leg on the shipped image is `ts_rank_cd`, not BM25**, and the report says so (B-073). Measured:
  `lexical_engine='bm25'` refuses with *"requires the pg_textsearch extension … not present on the shipped
  image"* — a clear refusal from the engine rather than a silent fallback. So a hybrid number from this image
  does not exercise the BM25 index that `load_documents` builds, and the docstring states that rather than
  letting a reader assume it.
- The documents table carries a **generated** tsvector column (B-073). Not a choice: `ts_rank_cd` takes a
  tsvector, so the column has to exist for the fusable leg to work at all. A comment claiming otherwise was
  written first and corrected by the measurement.

`rerank`, `vectorizer` and `ai_sql` remain undeclared for all three adapters, and that is the state rather than
an omission: each reaches an external model, and without an endpoint there is nothing to measure.

- **The lexical and Parquet pillars are reachable** (B-073), taking the TheoDB adapter from four declared
  capabilities to six. Both were found by reading `pg_proc` on the running server rather than the docs, and
  neither is where the documentation implied: `bm25_build` / `bm25_search` and `read_parquet` /
  `write_parquet` live in `public`, not in `theodb`. Verified against a real server — BM25 returns
  semantically correct ranks (the document containing both query terms scores 1.191, the two containing one
  each tie at 0.755), and 5 000 rows written with `write_parquet` read back through `read_parquet` with all
  three aggregations correct.
- A document load now carries **both legs from one corpus** (B-073): text for the lexical index and a vector
  for the dense one, in the same table. Loading only the text would make the hybrid surface unreachable from
  that corpus, and comparing two legs measured over different corpora compares the corpora.
- Searching a BM25 index that was never built is **refused** (B-073). It used to return zero rows, which is
  indistinguishable from nothing matching — the defect class this repository tracks as a surface answering
  where it should refuse.
- The Parquet directory is configuration, not a constant (B-073). The writer is the **server** process, so the
  path must be writable by the database user rather than by whoever runs the harness — measured the hard way,
  with `Permission denied (os error 13)` from a directory root had created inside the container.

- **A swept run emits its Pareto frontier** (B-067). `analysis/pareto.py` computed dominance and had no caller,
  so no run produced one — while the project's own rule says a headline throughput comparison needs a stated
  target quality with its interpolation method **or** the complete frontier, leaving every comparison to fall
  to the first branch by default. Each point records *which* configurations dominate it rather than only that
  it is dominated: an operator fixing one needs to know what to compare against. Below two measured
  configurations no frontier is written, because a frontier of one point is a point and publishing it as a
  curve would dress a single measurement as a trade-off.
- **`--baseline` runs regression detection** (B-067, B-072). `analysis/regression.py` implemented comparability
  checks, per-metric gates and a verdict, and had zero importers while three profiles declared
  `regression_gate = True`. First run against a real baseline returned **INCOMPARABLE** with the reason named:
  `profile: pr vs smoke`. That is the gate working — comparing a one-repetition run against a three-repetition
  one compares protocols, not code, and I22 requires the comparison to fail closed rather than report a
  reassuring number. Default thresholds are marked **advisory**, not measured: a regression threshold is only
  trustworthy once the runner's own variance has been characterised, and the same-day evidence for why they are
  not tighter is that the same configuration re-run on the same host varied by 24% and 46%.

- **The orchestrator depends on a workload protocol, not on one family** (B-067), so the analytical surface can
  finally be run. `RunRequest.workload` was typed `VectorWorkload` and the runner constructed `VectorBenchmark`
  by hand, which is why `bench/analytical.py`, `bench/graph.py` and `bench/retrieval.py` had zero importers in
  `src/` — 336 lines of analytical benchmark with its own oracle, the adapter methods and a four-state residency
  gate, and no way to invoke any of it. Five questions carried the coupling, and each is one a family should
  answer about itself: which benchmark to build, what the artefact says, how many operations were expected, how
  many were warm-up, and **what its quality axis even is**. That last one mattered most: the runner tested
  `recall is not None`, so an analytical run would have been invalidated for failing to produce a number that
  does not apply to it — approximate retrieval reports recall per repetition, an analytical answer is right or
  wrong once.
- `analytical/synthetic/paths` is a registered suite (B-067): the same seeded rows stored three ways, four
  aggregations against each, answers checked against the benchmark's own oracle. First VALID run, 200 000 rows
  on TheoDB: `total_rows` and `sum_amount` are **3.5× faster** on the columnar path, `filtered_sum` is 0.74×,
  and `group_by_category` is **0.18× — five times slower than heap**, which is the `GROUP BY` pushdown gap
  measured earlier now carried by a validated bundle rather than by a script. Parquet reports `not measured`
  on every query, because no adapter implements that path.
- **The TheoDB adapter now emits TheoDB's own access methods** (B-064). Measured against the image this
  project's own Dockerfile builds — PostgreSQL 18.6, `theodb_rs` 1.5.0 — the harness emitted
  `CREATE INDEX ... USING hnsw ("embedding" vector_l2_ops)` and the server answered
  `access method "hnsw" does not exist`. `pg_am` holds `theodb_hnsw` and `theodb_ivfflat`, with
  `theodb_hnsw_l2_ops` and friends. The bare `hnsw` name and the `vector_*_ops` classes do exist — in the
  separate `vector` compatibility shim, which the image creates in `template1` rather than in the
  `postgres` database a client reaches by default. So every indexed row of our own product's axis
  returned `INVALID`, while the exact-search row measured and was published: the bundle was not empty,
  it was partial. The engine's access-method name is now declared per adapter, while the bundle label
  stays the index family. Verified: the same run is `VALID` with a real recall curve
  (0.5928 → 0.7800 → 0.9650 across ef_search 16 / 64 / 256).
- The TheoDB adapter loads `theodb_rs` into the session (B-064). Measured: a fresh session holds zero
  `theodb%` rows in `pg_settings`, and no `hnsw.ef_search` either, until the LOAD runs — so every swept
  `ef_search` was a placeholder and the search ran at the default of 64. The LOAD is now issued by the
  base class for any adapter that declares a library, so a third engine cannot forget it.
- The server version reaches the bundle alongside the extension version, for every adapter that has an
  extension (B-064). One machine, one afternoon: TheoDB on PostgreSQL 18.6, pgvector on 17.11, AlloyDB
  Omni on 17.9. The comparison crosses a major version, and the only bundle that hid which PostgreSQL it
  ran on was our own product's, because its override replaced the base version instead of composing with
  it. Three separate implementations became one.
- Index parameters are rendered by type instead of forced through `int()` (B-059). Measured:
  `scann` accepts `quantizer='sq8'`, a string, and the previous renderer raised a bare
  `ValueError` with no phase, system or option name. Strings are now quoted and escaped through
  the existing literal helper; a type the renderer does not know is refused with an
  `AdapterError` rather than coerced, because a benchmark definition carrying a list where a
  scalar belongs is broken, and stringifying it would put an unintended index configuration into
  a published measurement.
- Operator classes are declared per adapter instead of by one shared table (B-059). Measured:
  the `scann` access method names its three classes `cosine`, `dot_product` and `l2` — none of
  pgvector's `vector_*_ops`. The lookup reads the table off the concrete class, so an adapter
  cannot inherit the wrong convention by accident.
- The contract test asserting that every adapter reports its effective search parameters is now
  parametrized off the registry instead of a written-out list (B-059). The list version was
  measured passing while `alloydbomni` was already registered and uncovered: a test that
  enumerates what it claims to cover universally excludes every adapter added after it was
  written, and reports green for doing so.
- A search parameter is now **verified in force before anything is measured** (B-060). The
  harness already refused to report a number when the planner ignored the index
  (`assert_index_used`); it did not refuse when the *knob* was ignored — the `SET` was issued
  and nothing read the value back. Measured on PostgreSQL 18: `SET nao.existe = 999` succeeds,
  `current_setting` hands back `999`, and `pg_settings` holds no such row. An unregistered
  namespaced GUC is accepted as a placeholder, so `current_setting` cannot detect it and
  `pg_settings` can. The gate reads `setting` and `source` from `pg_settings` and refuses when
  the GUC is absent, when the value diverges from what was sent, or when `source` is still
  `default`.
- The bundle records the search parameters **in force** alongside those requested, keyed by GUC
  name (B-060). The two are not always equal: `probes` is clamped to the list count, so a
  request of 10000 on a 10k-row table is sent as the clamp — and `points[].parameters` was built
  from the request, before the knobs were applied. No schema version changed: that field is
  already declared as an open object of scalars.
- `SystemAdapter.effective_search_parameters()` is part of the contract, and every registered
  adapter answers it — including `FakeAdapter`, which is the double the runner's own tests
  exercise most, so a contract that skipped it would be untested where it runs most (B-060).
- Versioned JSON schemas for every machine-readable artifact: benchmark,
  manifest, environment, dataset, system, validation, result, statistics,
  regression, pareto and summary. Artifacts are validated before being written,
  so an invalid file never lands in a bundle.
- `theodb-bench doctor`: fifteen host checks reporting PASS, WARN, FAIL or
  UNAVAILABLE. Which checks are mandatory depends on the profile, so a laptop
  can run a smoke benchmark and cannot produce a release claim.
- `theodb-bench env`: full environment capture from procfs and sysfs, with
  every undeterminable field recorded as an explicit absence carrying its
  reason.
- Immutable run bundles: finalization freezes the manifest and every raw
  measurement, while still allowing re-analysis to add new derived artifacts.
- Resource isolation with escape detection: a subprocess that leaves the
  declared CPU allocation is caught even on hosts where nothing could be
  enforced.
- Telemetry collectors (process, perf) that can be switched off and that
  measure their own overhead. A counter that could not be collected is recorded
  as absent, never as zero.
- Dataset layer identifying datasets by checksum: `dataset list`, `verify` and
  `fetch`, with atomic download and refusal to silently replace mismatched
  bytes.
- System adapter contract plus four adapters: a deterministic fake that
  produces nine real failure modes on demand, upstream PostgreSQL, pgvector and
  TheoDB.
- Vector ANN workload with untimed warm-up, per-configuration index isolation,
  query caps that appear in the label, and recall computed by the benchmark
  from its own oracle.
- Eleven-phase run orchestrator producing a complete, validated, immutable
  bundle.
- Analysis: recall by distance threshold following ANN-Benchmarks, nDCG, MRR,
  recall@n, latency percentiles, best-of-N throughput, aggregation that keeps
  every repetition, stability detection, Pareto frontiers and matched-quality
  selection.
- ANN dataset readers for ANN-Benchmarks HDF5 and the fvecs/ivecs family, and
  `theodb-bench run --dataset` to measure a verified corpus. Published
  distances are never read; recall recomputes them from the vectors.
- Reciprocal rank fusion, as an offline twin of the system's own fusion so the
  two can be compared rather than one trusted.
- Retrieval suite: lexical, dense, hybrid RRF and hybrid plus rerank over one
  corpus and one query set, reporting nDCG@10, Recall@k and MRR alongside
  throughput, with model latency in its own stage.
- Model endpoint abstraction (mock, local, remote) where only the deterministic
  mock may back a regression gate, and the mock's latency is required to be
  non-zero because an instant model changes the loop's concurrency regime.
- Operations suite measuring the foreground write clock and the
  time-to-freshness clock separately, across insert, update, backlog drain and
  worker saturation.
- Graph suite: 1/2/3-hop, BFS, fanout sweep, build and rebuild, with every
  traversal validated against an oracle before its timing is accepted.
- Analytical suite comparing row, columnar and Parquet execution on identical
  data, with per-stage timings and answer validation.
- Paired significance testing: randomisation test, bootstrap confidence
  interval and t-test cross-check, with Monte-Carlo correction and a fixed
  seed. Comparative significance claims are now possible rather than
  forbidden.
- Regression comparison that fails closed on an incomparable baseline and
  reports ADVISORY for any threshold not derived from a measured noise floor.
- Reports in both halves: a human report that leads with status and profile,
  and a machine summary carrying provenance and limitations.
- CI in two classes: shared correctness CI whose numbers are explicitly
  discarded, and a dedicated benchmark workflow that never triggers on a pull
  request.
- Methodology documents covering the measurement-integrity invariants and the
  agent workload surface.
- Agent workload is now the primary benchmark surface; the seven capability
  surfaces are components that explain an agent result rather than substitutes
  for it.
- Dataset manifests are JSON rather than YAML
  (`docs/decisions/0002-json-dataset-manifests.md`).

### Fixed
- A bulk dataset load no longer runs under the query time budget, which aborted a
  20 000 000-vector load partway through the COPY. Index build and bulk load now share
  one budget mechanism, because they are the same kind of unmeasured work (#B-073)
- The `filtered_sum` SQL filtered `quantity < 24` while the benchmark's oracle filters `category = 'a'`
  (B-067). It was copied from a published TPC-H-shaped query without reading the oracle that already existed,
  and **the harness caught it**: the run came back INVALID on both storage paths rather than reporting a fast
  wrong number. The oracle did exactly what it is for.

- **`theodb-bench head2head` measures two systems interleaved, query by query** (B-074). A paired test over two
  sequential runs removes the variance of query difficulty and leaves the variance of the machine: measured, the
  same configuration re-run on the same host varied by 24% and 46% in median throughput, and the paired test
  attributes that to the engine with the same confidence a real difference gets. Query *i* now goes to both
  systems back to back, **with the order alternating** — under a fixed order the first system pays the cold
  cache on every query and the second answers each one with the page cache just warmed, a bias
  indistinguishable from the second being faster.
  It changed a verdict. The sequential paired comparison reported TheoDB beating AlloyDB Omni with dz = -0.94
  on 448 of 500 queries; interleaved, Omni wins the mid-depth point and TheoDB wins only the deepest, both with
  small effects. Most of the sequential verdict was drift.
- Each side of a head-to-head declares **its own benchmark** (B-074). Two engines need different index
  configurations to reach the same quality and their knobs are not the same knobs — `pq_subspaces` is
  meaningless to AlloyDB, `num_leaves` is meaningless to us, and the first version of this command died on
  `unrecognized parameter "pq_bits"`. What must match is the experiment, and corpus, queries, k, metric and seed
  are checked rather than assumed.
- A head-to-head reports the **recall of both sides** and refuses a verdict when they differ by more than 0.01
  (B-074). A latency comparison between two operating points of different quality reports the one doing less
  work as faster. When a verdict is given and the recalls still differ, the side at lower recall is named, with
  how much of its advantage is work it did not do.
- Query counts are reported in the direction the verdict names (B-074). The effect field counts where the first
  system's value was *larger*, which for latency is where it was slower; printed beside "A beats B" it read as
  A losing on most queries.

- **Bulk load streams through binary COPY** (B-070), and the change is what makes a scale beyond a few million
  rows thinkable. Measured on one million SIFT-128 vectors, same host: batched `executemany` **122 s** →
  text COPY **75 s** → binary COPY **16.8 s**, a 7.3× improvement end to end. The middle step is why the last
  one exists: of the 75 s a text COPY took, **72 were the Python text encoding** — one `repr()` per float,
  128 million of them — so cutting round-trips had already given everything it could. The binary encoder is
  numpy writing the whole block in one pass, with no per-value Python at all. Verified against a real server:
  1 000 000 rows loaded, and five vectors spread across the corpus including both ends compared element by
  element with zero divergence.
- PostgreSQL's binary COPY layout is written out in `copy_binary.py` rather than delegated to a per-type dumper
  (B-070): psycopg has no binary dumper for `vector`, and adding one means a dependency that helps only the
  pgvector-family adapters. Hand-writing a wire format is safe only if it is pinned, so every field — the
  signature, the flags, the per-row field count, each field's length prefix, and pgvector's own
  `int16 dim, int16 unused, big-endian float4` — is asserted byte for byte. Upstream PostgreSQL keeps the text
  path: `real[]` has a different and more elaborate array layout, and exact search over it is the honest floor
  of a comparison rather than a scale target.

- **Comparing two systems runs a paired test, or says it cannot** (B-071). Invariant I14 requires a paired
  test rather than a comparison of means, and `analysis/significance.py` implemented one — paired
  randomisation, paired bootstrap CI, Cohen's dz, Monte-Carlo correction — with **zero importers** in `src/`,
  while `compare` put medians in adjacent columns. The missing link was upstream of both: the measurement loop
  collected a latency per query and kept only the summary, so no bundle carried the paired samples. Bundles now
  record `latency-by-query.json`, and `compare` reports direction, `p`, a 95% CI and effect size per shared
  configuration. Samples are keyed by **query id, never by position**: the loop skips queries that errored or
  timed out, so position *i* is not query *i*, and pairing by position would misalign every sample after the
  first timeout and produce a confident wrong verdict. Differing query sets are refused rather than
  intersected — an intersection tests the subset where both systems succeeded, an easier question that favours
  whichever system dropped its hardest queries. Repetitions are averaged per query before pairing, because a
  query measured three times on each side is one paired observation, not three.
- Two engines are paired **at matched recall**, never by configuration label (B-071). Their labels carry
  engine-specific knobs — `probes=20` on one side, `num_leaves_to_search=20` on the other — and two knobs named
  differently and set to the same integer are not the same operating point. Quality is the axis both share, so
  the frontiers are read there, with a tolerance that is honoured rather than taking the nearest pair at any
  distance: frontiers that never meet have no comparable point, and inventing one would compare a fast
  low-quality configuration against a slow high-quality one and report the first as a winner. Same-label
  pairing is kept for the regression case, where it is the right key.
- The paired verdict states what the pairing does **not** control (B-074). It removes the variance of query
  difficulty; it does not remove drift in the machine, because the two runs happen at different times. Measured
  the same day: the same configuration re-run on the same host varied by 24% and 46% in median throughput, so a
  busier machine during one side is attributed to the engine with the same confidence a real difference would
  be — and the confidence interval does not protect against it, because it measures dispersion across queries
  rather than across runs.
- Two profile flags that promised gates now run them (B-072). Measured: of the four rigour flags a profile
  declares, `publishable` and `dirty_tree_invalidates` steered code and `regression_gate` and
  `frozen_methodology` steered nothing — both appeared only as values echoed by `theodb-bench list`. They now
  emit real checks, with the semantics invariant I22 states rather than a stricter invention: a run with no
  baseline records that no regression detection happened (`UNAVAILABLE`), and a baseline that is **not
  comparable** fails closed. Demanding a baseline of every run would have made the gate impossible to adopt —
  the first run of any suite has nothing to regress against — and the first draft of this change did exactly
  that, invalidating 68 tests' worth of runs before the semantics were corrected.
- A test that asserts every profile flag steers something (B-072), by AST rather than by grep. Textual presence
  is not consumption: both decorative flags *did* appear in `src/`, as dict values in a printed listing. The
  test looks for the flag in a condition, a boolean operator, a negation, or a `required=` argument.

- **An aborted run says why it aborted, instead of blaming the system under test** (B-057). The runner set
  `sut_crashed = True` for any exception, so three different facts reached the report as one sentence — "system
  under test crashed during the run". Measured in a single session on 2026-08-17: the knob gate refusing a run
  because a parameter was not in force (the harness working as designed), a `CREATE INDEX` cancelled by the
  harness's *own* 60 s `statement_timeout`, and no crash at all — the container was `Up (healthy)` with no PANIC
  and no FATAL in its log. For results that get published, that misattribution is the expensive kind: a reader
  who sees the system crashed concludes the database is unstable. Aborts are now classified into `sut_alive`,
  `run_not_refused` and `within_time_budget`, all three required, each with its own remediation. Classification
  uses the driver's real exception classes in an order the tests pin, because `psycopg.errors.QueryCanceled` is
  a subclass of `psycopg.OperationalError` — checking connection-loss first would have called every cancelled
  statement a crash, which is the same misattribution one layer down. An unrecognised abort is reported as a
  crash: the conservative direction, since hiding a real crash behind a harness message is the failure being
  removed, pointed the other way.
- **Building an index has its own time budget** (B-057), separate from the query budget and restored afterwards
  even when the build fails. Measured: an hnsw build over one million SIFT-128 vectors was cancelled at 61 s
  under the 60 s query budget, while the competitor's scann build fitted inside it — so one shared budget
  silently decided which engines were measurable at which scale, and the report blamed the engine. A k=10 query
  taking a minute is still a defect worth catching, so the query budget stays tight at 60 s; the build gets an
  hour, generous enough for a billion-scale build and still short enough to catch a hung one.
- **TheoDB's own ScaNN-class path is reachable from the harness** (B-057). TheoDB has the ScaNN recipe as
  reloptions on `theodb_ivfflat` rather than as an access method named `scann`: `pq_subspaces` is the
  anisotropic quantizer, `pq_bits=4` the LUT16 width, `aq_threshold` ScaNN's anisotropic T, `soar_lambda` its
  SOAR spilling, and `separate_storage=1, refine=1` the exact-distance second stage. The internal name for the
  arc is pg_scann. The rescore pool is `64 * theodb_hnsw.over_fetch`, and that knob is now declared and swept —
  without it the harness could only sweep probe depth, which measures a quantized index at its quantizer's
  fidelity rather than at its operating point.
- `vector/sift/pg-scann` is registered as the counterpart of `vector/sift/scann-ah` (B-057), with both stages
  matched: probe depth against `num_leaves_to_search`, and a rescore pool of 128 against their 100 — the closest
  the two knobs reach, recorded in the artefact so a reader can see they are not identical.

- **SIFT1M is a registered dataset** (B-057): one million 128-dimensional SIFT descriptors with 10 000 queries,
  identified by the checksum of the bytes actually downloaded
  (`dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`, 525 128 288 bytes) rather than by a
  version string. It is the corpus ADR-0035 used when it measured the ~25x QPS gap against the ScaNN
  *library*, so it is the only corpus on which a measurement against the scann *access method* can be
  compared to that conclusion. The licence is recorded as unverified and `redistributable: false`, because
  the TEXMEX/INRIA corpus repackaged by ANN-Benchmarks carries no licence text this project could confirm —
  and inventing one to fill the field is the fabrication the manifest exists to prevent.
- **The ScaNN rerank depth is a declared, swept search knob** (B-057), and without it an AH frontier measures
  quantization error rather than the index. Measured at 100 000 SIFT-128 vectors with `quantizer=AH` and 80 of
  316 leaves searched: `scann.pre_reordering_num_neighbors` ships at `-1`, and at that default recall@10 is
  **0.6568**; at 100 it is **0.9964**; at 500, **0.9998**. Searching four times more leaves had bought 1.4
  points, so the ceiling was the missing exact-distance rescore, which is how ScaNN is designed to work. A
  frontier taken at the default would have reported the competitor topping out at two thirds recall — false,
  and it happened to flatter us.
- **AH quantization is applied at build time, and verified in force** (B-057). Measured:
  `CREATE INDEX ... WITH (quantizer='AH')` fails with `AH quantization is not enabled for the index` unless
  `scann.enable_ah_quantizer` is on in the building session; valid quantizer values are `SQ8`, `Flat` and `AH`,
  and the flag ships off. Build-time settings are therefore a separate declaration from the search knobs: one
  applied after the index exists changes nothing about the index already written, so an adapter treating one as
  the other builds SQ8 and labels it AH.
- **A scann search sweep is a registered suite** (B-057): `vector/synthetic/scann-sweep` sweeps
  `num_leaves_to_search` over a scann index, as the hnsw suite sweeps `ef_search`. They are separate suites
  because the search knob belongs to the index family and one suite cannot ask for both — an adapter that
  cannot apply a requested knob now refuses the run. The two families are compared at **matched recall** from
  their frontiers, never by pairing knob values that mean different things.
- **The analytical surface reaches real engines** (B-061): `load_analytical` and `execute_analytical` are
  implemented for the PostgreSQL family. Before this, only the in-process fake could execute an analytical
  query — the 336-line `AnalyticalBenchmark`, its oracle and its three declared execution paths had no
  engine attached to them. The heap path is a plain table; TheoDB's columnar path is
  `CREATE TABLE ... USING theodb_columnar`; AlloyDB Omni's is a heap table plus a cache registration. Two
  mechanisms under one label, declared per adapter rather than derived from the label.
- **A residency gate that refuses every state in which the columnar label would be a lie** (B-061).
  Measured against a running AlloyDB Omni, its columnar engine has four distinguishable states and three of
  them answer queries correctly while silently falling back to heap: engine off (the default, and its
  context is `postmaster`, so only a restart changes it); enabled but never populated; **enabled and
  registered with an empty store**; and enabled, populated, actually used. The third one is why the gate is
  not built on `g_columnar_columns`, which the published independent evaluation recommends as the residency
  proof: measured, that view reported **4 columns while the engine summary reported Memory Used = 0 MB**,
  and the plan was still a sequential scan. It reports registration, not residency. The measured cause is
  that the refresh needs shared memory a default Docker container does not have — it fails with
  `could not resize shared memory segment`. Each state produces a different message, because they need
  different actions.
- The columnar aggregate pushdown is enabled and **verified in force** before an analytical number is taken
  (B-061). `theodb.enable_columnar_agg` ships off. Measured on the built image, same table of one million
  rows and same query: off → `Seq Scan`, **1407 ms**; on → `Custom Scan (theodb_columnar_agg)`, **108 ms**.
  Thirteen times, decided by a GUC, with the catalog reporting a columnar table either way. Leaving it at
  the default measures columnar storage without its pushdown — a path the project already knows loses to
  heap — and publishing that as "our columnar" would be the same error as measuring ScaNN with its AH
  quantizer off.
- The plan proof is taken **per query, not per table** (B-061), because pushdown coverage depends on the
  query shape. Measured at one million rows with the pushdown on: `sum(amount)` plans as
  `Custom Scan (theodb_columnar_agg)`, while `GROUP BY category` falls back to Seq Scan → external-merge
  Sort (25 456 kB spilled to disk) → GroupAggregate and runs **14× slower than heap**. A gate that probed
  one query and generalised would have called the grouped one pushed down.
- **AlloyDB Omni is a measurable system** (B-059): `alloydbomni` is a registered adapter driving
  Google's `scann` access method. Every property below was measured against
  `google/alloydbomni:latest` on an ephemeral droplet rather than read from documentation.
  `capabilities()` declares only what this code exercises, and deliberately claims none of the
  managed service's platform features — Omni is a query layer, with no disaggregated storage,
  read pool or managed failover. Racing against capabilities the running product does not have
  would measure a product that is not there.
- The adapter issues `LOAD 'alloydb_scann'` per session, and the knob gate from B-060 is what
  proves it took effect (B-059). Measured: in a session without the LOAD,
  `SET scann.num_leaves_to_search = 500` **succeeds**, `current_setting` echoes `500` back, and
  `pg_settings` does not list the GUC — the engine keeps searching at its default of 0.
  `shared_preload_libraries` does not carry the library. This is the same placeholder mechanism
  the gate was built for, now confirmed on another engine; removing the LOAD makes two tests
  fail rather than producing a shallow result.
- The measured server version reaches the bundle read from the server, never inferred from the
  image tag (B-059). The published image serves **PostgreSQL 17.9**, so a head-to-head against
  TheoDB crosses a major version — a fact a report has to state, not hide. A server that will
  not answer gets no version invented for it: the field is omitted.

- **A requested search knob the adapter cannot apply is now refused** (B-059). The gate added in
  B-060 verified every knob it *mapped* and silently accepted every knob it did not — and a second
  engine is what exposed the difference. Measured against a running AlloyDB Omni: its bundled
  pgvector fork registers no `hnsw.*` GUC at all (zero rows in `pg_settings`), and the Omni adapter
  maps `num_leaves_to_search`, not `ef_search`. A sweep of `ef_search` therefore produced an empty
  mapping, the gate had nothing to check, and it passed vacuously: recall measured **0.7820 at both
  ef_search=16 and ef_search=256**, and the bundle published three rows labelled 16 / 64 / 256 that
  were one operating point. Each adapter now declares the knobs it understands, and a request naming
  anything else fails the run instead of relabelling a default. The same command that produced the
  fictional rows now reports `INVALID`.
- **The recall oracle can no longer be OOM-killed by the corpus it is supposed to measure** (B-057).
  `brute_force_ground_truth` was measured being killed at **10.5 GB** of resident memory while building ground
  truth for a 512 MB corpus — one million 128-dimensional vectors against 500 queries, on a 16 GB host. The
  cause was two allocations, neither algorithmic: a `(queries x corpus)` float64 distance matrix (4 GB) and an
  identically shaped int64 tile of `arange(corpus_size)` (4 GB) whose only purpose was breaking ties by id. It
  also full-sorted a million distances per query to take the top ten. It now chunks over queries, keeping the
  working matrix in the tens of megabytes whatever the corpus size, and selects with a partition plus a
  tie-safe re-admission of everything level with the k-th element. Behaviour is unchanged and pinned by tests
  that compare against the previous implementation across all three metrics, including the case a careless
  partition gets wrong: ties spanning the top-k boundary must still resolve by ascending id, or recall stops
  being reproducible. A benchmark harness that cannot build ground truth at the scale it is meant to measure
  is not measuring that scale, and the tests assert the memory ceiling rather than a wall time.

- The module docstring of `src/adapters/postgres.py` no longer claims an invariant the code does not
  enforce. It advertised, as I5, that "the index is forced *and* verified"; measured 2026-08-17,
  `assert_index_used` has no caller anywhere in the package, raises `ProgrammingError` if called
  (this class overrides `_query_sql` to repeat the distance expression, so the probe binds twice
  while the inherited verifier binds once), and `SET enable_seqscan = off` appears in that docstring
  and nowhere else in executable code. The harness measures whatever plan the planner picks. No
  published number is retracted: at the registered suite's size (10 000 × 64) EXPLAIN confirms the
  planner does choose the index on pgvector, Omni/hnsw and Omni/scann — but at 200 rows it chose a
  sequential scan, so the hole is latent rather than harmless. The mechanism is tracked separately;
  what changed here is that the file stops asserting something untrue.
- A run manifest could name a dataset the run never measured: `dataset_id` was
  recorded while the workload generated a synthetic corpus. Declaring a dataset
  now requires supplying the vectors, and supplying vectors requires declaring
  their identity.
- The TheoDB adapter declared hybrid, lexical, columnar, Parquet, graph and
  vectorizer capabilities it does not implement, putting false claims into
  every `system.json`. It now declares only the vector surface it can exercise.

[Unreleased]: https://github.com/usetheoai/theodb-bench
