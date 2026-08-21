"""The CLI is the only surface most users touch; its exit codes are a contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from theodb_bench.cli import EXIT_ERROR, EXIT_OK, EXIT_PREFLIGHT_BLOCKED, main


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == EXIT_OK
    assert "theodb-bench" in capsys.readouterr().out


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != EXIT_OK


def test_doctor_smoke_runs_and_reports(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--profile", "smoke"])
    out = capsys.readouterr().out
    assert code in {EXIT_OK, EXIT_PREFLIGHT_BLOCKED}
    assert "theodb-bench doctor" in out
    assert "pass," in out


def test_doctor_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    main(["doctor", "--profile", "pr", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "pr"
    assert isinstance(payload["checks"], list)
    assert {"id", "outcome", "detail", "required"} <= set(payload["checks"][0])


def test_doctor_exit_code_reflects_permission(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--profile", "smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == (EXIT_OK if payload["may_run"] else EXIT_PREFLIGHT_BLOCKED)


def test_doctor_rejects_an_unknown_profile() -> None:
    with pytest.raises(SystemExit):
        main(["doctor", "--profile", "production"])


def test_env_json_matches_the_environment_schema(capsys: pytest.CaptureFixture[str]) -> None:
    from theodb_bench.schemas import validate

    assert main(["env", "--json"]) == EXIT_OK
    validate("environment", json.loads(capsys.readouterr().out))


def test_env_human_output_never_prints_a_blank_or_bare_zero_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Every line is "field   value". A field the host could not answer must
    # render as <unavailable: reason>, never as an empty string and never as a
    # bare 0 that would read as a measurement.
    assert main(["env"]) == EXIT_OK
    for line in capsys.readouterr().out.splitlines():
        field, _, value = line.partition("  ")
        rendered = value.strip()
        assert rendered, f"{field} rendered empty"
        assert rendered != "0", f"{field} rendered as a bare zero"
        if "<" in rendered:
            assert ">" in rendered, f"{field} has an unterminated absence marker"


def test_profiles_lists_publishability(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profiles"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "release" in out
    assert "not publishable" in out


def test_profiles_json_exposes_the_rules(capsys: pytest.CaptureFixture[str]) -> None:
    main(["profiles", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["release"]["publishable"] is True
    assert payload["smoke"]["publishable"] is False


def test_schema_list_names_every_schema(capsys: pytest.CaptureFixture[str]) -> None:
    from theodb_bench.schemas import SCHEMA_NAMES

    assert main(["schema", "list"]) == EXIT_OK
    listed = capsys.readouterr().out.split()
    assert listed == list(SCHEMA_NAMES)


def test_schema_validate_accepts_a_valid_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    fixture = Path(__file__).parent / "fixtures" / "manifest.json"
    assert main(["schema", "validate", "manifest", str(fixture)]) == EXIT_OK
    assert "valid against manifest schema" in capsys.readouterr().out


def test_schema_validate_reports_a_broken_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert main(["schema", "validate", "manifest", str(broken)]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_typed_errors_become_messages_not_tracebacks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["schema", "validate", "manifest", str(tmp_path / "missing.json")]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


# ---------------------------------------------- raising the budget the error names
#
# When a bulk phase exceeds its budget the harness says: "Raise the budget for
# this phase or reduce the scale". Until now it offered no way to do the first,
# which makes the advice a dead end at exactly the scale that needs it — a
# 20 000 000-vector HNSW build is hours, and the default budget is one.


def test_the_run_command_accepts_a_bulk_budget() -> None:
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(
        ["run", "vector/synthetic/smoke", "--system", "fake", "--build-timeout", "7200"]
    )

    assert args.build_timeout == 7200


def test_the_bulk_budget_is_optional_and_defaults_to_the_adapter_setting() -> None:
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(["run", "vector/synthetic/smoke", "--system", "fake"])

    assert args.build_timeout is None


def test_the_bulk_budget_reaches_the_adapter() -> None:
    """The flag has to change the adapter the run actually uses; a flag parsed
    and dropped is worse than no flag, because the run then looks configured."""
    from theodb_bench.adapters.postgres import PgvectorAdapter
    from theodb_bench.cli import adapter_overrides

    overrides = adapter_overrides(build_timeout=7200)
    adapter = PgvectorAdapter(**overrides)

    assert adapter.config.build_timeout_ms == 7_200_000


def test_no_flag_leaves_the_adapter_default_untouched() -> None:
    from theodb_bench.adapters.postgres import PgvectorAdapter
    from theodb_bench.cli import adapter_overrides

    assert adapter_overrides(build_timeout=None) == {}
    assert PgvectorAdapter().config.build_timeout_ms == 3_600_000


def test_a_negative_budget_is_refused() -> None:
    """A zero or negative statement_timeout means *no limit* in PostgreSQL, so
    accepting one here would silently remove the guard rather than widen it."""
    from theodb_bench.cli import adapter_overrides

    with pytest.raises(Exception, match="positive"):
        adapter_overrides(build_timeout=0)


# --- isolamento declarado pela CLI -------------------------------------------------
#
# MEDIDO em 2026-08-21, rodando o arnes num host dedicado: os perfis `nightly` e `release`
# declaram `isolation_required`, e com ele `cpu_limit` e `memory_limit` viram checks OBRIGATORIOS.
# A CLI nunca construia um `IsolationPlan`, entao `RunRequest.isolation` ficava no default vazio,
# os dois checks saiam `UNAVAILABLE` e a corrida era `INVALID` — em QUALQUER hardware.
#
# Ou seja: dois dos cinco perfis do arnes eram inalcancaveis pelo seu proprio ponto de entrada.
# Nao e limitacao de maquina; e superficie faltando.


def test_a_cli_aceita_declaracao_de_cpu_set() -> None:
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(
        ["run", "vector/synthetic/smoke", "--system", "fake", "--cpu-set", "2-5"]
    )

    assert args.cpu_set == "2-5"


def test_a_cli_aceita_declaracao_de_limite_de_memoria() -> None:
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(
        ["run", "vector/synthetic/smoke", "--system", "fake", "--memory", "8GiB"]
    )

    assert args.memory == "8GiB"


def test_o_isolamento_declarado_CHEGA_ao_plano() -> None:
    """Uma flag lida e descartada e pior que nenhuma flag — a corrida passa a parecer
    configurada e o portao continua reprovando sem que ninguem entenda por que.

    Este teste existe porque o defeito original era exatamente esse: o plano nunca
    era construido, e o unico sintoma visivel era um `INVALID` no fim da corrida.
    """
    from theodb_bench.cli import build_isolation_plan
    from theodb_bench.isolation import parse_cpu_set

    plano = build_isolation_plan(cpu_set="2-5", memory="8GiB", numa_node=None)

    assert plano.cpu_set == parse_cpu_set("2-5")
    assert plano.memory_bytes == 8 * 1024**3


def test_um_plano_sem_declaracao_permanece_vazio() -> None:
    """Nao declarar continua sendo legitimo: o perfil `research` nao exige isolamento,
    e inventar um default esconderia do usuario que nada foi declarado."""
    from theodb_bench.cli import build_isolation_plan

    plano = build_isolation_plan(cpu_set=None, memory=None, numa_node=None)

    assert plano.cpu_set is None
    assert plano.memory_bytes is None
