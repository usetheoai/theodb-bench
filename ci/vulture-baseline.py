# Baseline de código morto do arnês — gerado, não escrito à mão.
#
# POR QUE EXISTE (B-063). O `assert_index_used` sobreviveu SEM CHAMADOR sendo citado por outro
# item como "a disciplina exata", e nada no repositório podia notar: não havia detector nenhum.
# Ao ligar um, a medição decidiu a configuração em vez do gosto:
#
#   · confiança >= 80  ->  4 achados, todos parâmetros de `__exit__` exigidos pelo protocolo
#   · confiança >= 60  ->  71 achados, dos quais 13 são MÉTODOS sem chamador
#
# E a pergunta que decidiu: **em qual confiança o vulture teria pego o defeito do B-063?**
# Medido por simulação (removendo o chamador recém-criado e re-rodando): `unused method
# 'assert_index_used' (60% confidence)`. Em 80 o portão passaria limpo sobre exatamente a
# classe que ele existe para pegar — seria teatro. Roda em 60.
#
# CONTRATO: este arquivo faz BASELINE dos 71 que já existiam. O portão reprova o que for NOVO.
# Zerar de uma vez, com 13 métodos a investigar um a um, é o mutirão que a lição do baseline de
# clippy do theo-db diz para não tentar. SUNSET 2026-11-20 (90 dias) — cada entrada sai conforme
# o símbolo é removido ou ganha chamador, e o que sobrar na data volta a reprovar.
#
# Regenerar:  python3 -m vulture src/ --min-confidence 60 --make-whitelist > ci/vulture-baseline.py
# Regenerar é um ATO DELIBERADO, não um passo de rotina: regenerar depois de introduzir código
# morto apaga a evidência em vez de reagir a ela.

_.invalidates  # unused property (src/abort.py:98)
invalid  # unused function (src/absent.py:71)
METRICS  # unused variable (src/adapters/base.py:61)
search_parameters  # unused variable (src/adapters/base.py:111)
search_parameters  # unused variable (src/adapters/base.py:129)
parameters_in_force  # unused variable (src/adapters/base.py:161)
accepted  # unused variable (src/adapters/base.py:247)
edge_type  # unused variable (src/adapters/base.py:256)
edge_type  # unused variable (src/adapters/base.py:267)
exc_type  # unused variable (src/adapters/base.py:605)
tb  # unused variable (src/adapters/base.py:607)
_.knn_sql  # unused method (src/adapters/postgres.py:675)
_.embed  # unused method (src/ai.py:102)
_.generate_batch  # unused method (src/ai.py:105)
MockEndpoint  # unused class (src/ai.py:115)
_.generate_batch  # unused method (src/ai.py:175)
_.embed  # unused method (src/ai.py:198)
LocalEndpoint  # unused class (src/ai.py:215)
_.embed  # unused method (src/ai.py:264)
RemoteEndpoint  # unused class (src/ai.py:271)
_.embed  # unused method (src/ai.py:325)
CostBreakdown  # unused class (src/ai.py:332)
require_gate_eligible  # unused function (src/ai.py:373)
rank_agreement  # unused function (src/analysis/fusion.py:70)
matched_quality  # unused function (src/analysis/pareto.py:122)
recall_from_ids  # unused function (src/analysis/quality.py:263)
success_at_k  # unused function (src/analysis/quality.py:322)
gates_from_noise_floor  # unused function (src/analysis/regression.py:202)
throughput_best_of_n  # unused function (src/analysis/statistics.py:102)
_.compare_paths  # unused method (src/bench/analytical.py:401)
timed_reference_scan  # unused function (src/bench/analytical.py:441)
GraphBenchmark  # unused class (src/bench/graph.py:200)
rebuild_delta  # unused function (src/bench/graph.py:341)
timed_reference_traversal  # unused function (src/bench/graph.py:357)
OperationsBenchmark  # unused class (src/bench/operations.py:157)
compare_clocks  # unused function (src/bench/operations.py:365)
RetrievalBenchmark  # unused class (src/bench/retrieval.py:239)
_.run_pipeline  # unused method (src/bench/retrieval.py:265)
_.offline_fusion  # unused method (src/bench/retrieval.py:379)
_.operation_count  # unused property (src/bench/vector.py:148)
_.synthetic  # unused attribute (src/bench/vector.py:373)
_.synthetic  # unused attribute (src/bench/vector.py:378)
_._ground_truth_ids  # unused attribute (src/bench/vector.py:389)
_._ground_truth_ids  # unused attribute (src/bench/vector.py:393)
_.tenant_of  # unused method (src/bench/vector.py:395)
MANIFEST_SCHEMA_VERSION  # unused variable (src/bundle.py:30)
RUN_ID_PATTERN  # unused variable (src/bundle.py:33)
_.append_raw_jsonl  # unused method (src/bundle.py:237)
_.raw_files  # unused method (src/bundle.py:294)
DATASET_SCHEMA_VERSION  # unused variable (src/datasets.py:35)
redistributable  # unused variable (src/datasets.py:78)
preprocess_version  # unused variable (src/datasets.py:80)
properties  # unused variable (src/datasets.py:83)
ENVIRONMENT  # unused variable (src/errors.py:20)
WARMUP  # unused variable (src/errors.py:25)
COOLDOWN  # unused variable (src/errors.py:27)
RunValidationError  # unused class (src/errors.py:164)
ComparabilityError  # unused class (src/errors.py:172)
read_fvecs  # unused function (src/formats.py:149)
read_ivecs  # unused function (src/formats.py:203)
interleaved  # unused variable (src/interleaved.py:60)
unreadable_processes  # unused function (src/isolation.py:208)
exc_type  # unused variable (src/isolation.py:260)
tb  # unused variable (src/isolation.py:262)
assert_no_escapes  # unused function (src/isolation.py:403)
scheduled_seconds  # unused variable (src/load.py:88)
default_cpu_plan  # unused function (src/runner.py:575)
peak_rss_bytes  # unused variable (src/telemetry.py:124)
voluntary_switches  # unused variable (src/telemetry.py:126)
involuntary_switches  # unused variable (src/telemetry.py:127)
is_present_bool  # unused function (src/validation.py:439)


# ============================================================================
# LIMITACOES ESTRUTURAIS DA FERRAMENTA — sem sunset, e a distincao importa.
# ============================================================================
# As entradas ACIMA sao DIVIDA: codigo morto real, com sunset em 2026-11-20, para sair conforme cada
# simbolo e removido ou ganha chamador. As abaixo sao outra coisa: casos em que o vulture esta
# narrowly certo e a conclusao errada.
#
# Um MEMBRO DE ENUM existe para ser escolhido pelo usuario, nao referenciado em codigo. `Regime` tem
# dois valores e a CLI os oferece por `choices=[r.value for r in Regime]`; o membro que nao e o
# default nunca aparece por nome dentro de `src/`. Escrever uma referencia para satisfazer o linter
# seria pior que a entrada aqui — seria codigo cuja unica razao de existir e um falso positivo.
#
# Misturar as duas classes num arquivo so faria a data de sunset mentir sobre metade dele.
Regime.EXCEEDS_CACHE  # unused attribute (src/bench/contention.py:38) — membro de enum
