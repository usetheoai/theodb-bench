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
