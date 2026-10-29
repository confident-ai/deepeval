"""`deepeval set-nace` / `unset-nace` / `set-typesafe`: the System One
provider (`DEEPEVAL_DECISION_PROVIDER`) and Nace settings they persist."""

from pathlib import Path
from typing import Dict

from typer.testing import CliRunner

from deepeval.cli.main import app as cli_app
from deepeval.config.eval_mode import (
    DecisionProvider,
    normalize_decision_provider,
    resolve_decision_provider,
)
from deepeval.config.settings import reset_settings


def _read_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'\"")
    return env


def test_set_nace_selects_nace_and_persists(env_path: Path):
    runner = CliRunner()
    save = f"dotenv:{env_path}"

    result = runner.invoke(
        cli_app,
        [
            "set-nace",
            "--model",
            "drex-max",
            "--base-url",
            "https://openrouter.ai/api",
            "--save",
            save,
        ],
    )

    assert result.exit_code == 0, result.output
    env = _read_dotenv(env_path)
    assert env["DEEPEVAL_DECISION_PROVIDER"] == "nace"
    assert env["NACE_MODEL_NAME"] == "drex-max"
    assert env["NACE_BASE_URL"].startswith("https://openrouter.ai/api")
    assert resolve_decision_provider() is DecisionProvider.NACE


def test_unset_nace_returns_to_typesafe(env_path: Path):
    runner = CliRunner()
    save = f"dotenv:{env_path}"
    runner.invoke(cli_app, ["set-nace", "--model", "drex-max", "--save", save])

    result = runner.invoke(cli_app, ["unset-nace", "--save", save])

    assert result.exit_code == 0, result.output
    env = _read_dotenv(env_path)
    assert "DEEPEVAL_DECISION_PROVIDER" not in env
    assert "NACE_MODEL_NAME" not in env
    assert resolve_decision_provider() is DecisionProvider.TYPESAFE


def test_set_typesafe_switches_back_from_nace(env_path: Path):
    runner = CliRunner()
    save = f"dotenv:{env_path}"
    runner.invoke(cli_app, ["set-nace", "--save", save])

    result = runner.invoke(cli_app, ["set-typesafe", "--save", save])

    assert result.exit_code == 0, result.output
    assert _read_dotenv(env_path)["DEEPEVAL_DECISION_PROVIDER"] == "typesafe"
    assert resolve_decision_provider() is DecisionProvider.TYPESAFE


def test_set_typesafe_alone_adds_no_provider_setting(env_path: Path):
    runner = CliRunner()

    result = runner.invoke(
        cli_app,
        [
            "set-typesafe",
            "--model",
            "jev-latest",
            "--save",
            f"dotenv:{env_path}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DEEPEVAL_DECISION_PROVIDER" not in _read_dotenv(env_path)


def test_eval_mode_message_names_the_configured_provider(env_path: Path):
    runner = CliRunner()
    save = f"dotenv:{env_path}"
    runner.invoke(cli_app, ["set-nace", "--save", save])

    result = runner.invoke(
        cli_app, ["set-eval-mode", "system_one", "--save", save]
    )

    assert result.exit_code == 0, result.output
    assert "Drex" in result.output
    assert "NACE_API_KEY" in result.output


def test_decision_provider_setting_is_lenient(monkeypatch):
    assert normalize_decision_provider(" NACE ") is DecisionProvider.NACE
    assert normalize_decision_provider("drex") is None

    monkeypatch.setenv("DEEPEVAL_DECISION_PROVIDER", "unknown")
    reset_settings(reload_dotenv=False)
    assert resolve_decision_provider() is DecisionProvider.TYPESAFE
