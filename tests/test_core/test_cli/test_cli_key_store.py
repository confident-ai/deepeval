import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from deepeval.cli.main import app
from deepeval.key_handler import (
    KEY_FILE_HANDLER,
    ModelKeyValues,
    EmbeddingKeyValues,
)

HIDDEN_DIR = ".deepeval"
KEY_FILE = ".deepeval"


def _read_hidden_store(tmp_path: Path) -> dict:
    store = tmp_path / HIDDEN_DIR / KEY_FILE
    if not store.exists():
        return {}
    try:
        return json.loads(store.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def test_cli_set_azure_does_not_persist_secrets_to_hidden_store(tmp_path: Path):
    """set-azure-openai should NOT write API key to .deepeval/.deepeval (JSON)."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "set-azure-openai",
            "--openai-endpoint",
            "https://fake-endpoint.openai.azure.com",
            "--openai-api-key",
            "sk-should-not-be-persisted",
            "--deployment-name",
            "fake-deployment",
            "--openai-api-version",
            "2024-01-01",
            "--openai-model-name",
            "gpt-4o-mini",
            "--model-version",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    data = _read_hidden_store(tmp_path)

    # non-secret config toggles should exist
    assert data.get(ModelKeyValues.AZURE_MODEL_NAME.value) == "gpt-4o-mini"
    assert (
        data.get(ModelKeyValues.AZURE_OPENAI_ENDPOINT.value)
        == "https://fake-endpoint.openai.azure.com/"
    )
    assert data.get(ModelKeyValues.OPENAI_API_VERSION.value) == "2024-01-01"
    assert (
        data.get(ModelKeyValues.AZURE_DEPLOYMENT_NAME.value)
        == "fake-deployment"
    )
    assert data.get(ModelKeyValues.AZURE_MODEL_VERSION.value) == "1"
    assert data.get(ModelKeyValues.USE_AZURE_OPENAI.value) == "YES"

    # secret must NOT be persisted
    assert ModelKeyValues.AZURE_OPENAI_API_KEY.value not in data


def test_key_handler_refuses_secret_write(tmp_path: Path, monkeypatch):
    """Directly ensure KEY_FILE_HANDLER refuses to persist blacklisted secrets."""
    monkeypatch.chdir(tmp_path)
    # Attempt to persist a secret directly (simulates accidental call)
    KEY_FILE_HANDLER.write_key(
        ModelKeyValues.AZURE_OPENAI_API_KEY, "sk-should-not-write"
    )
    KEY_FILE_HANDLER.write_key(
        ModelKeyValues.DEEPSEEK_API_KEY, "ds-should-not-write"
    )

    data = _read_hidden_store(tmp_path)
    assert ModelKeyValues.AZURE_OPENAI_API_KEY.value not in data
    assert ModelKeyValues.DEEPSEEK_API_KEY.value not in data


def test_unset_azure_clears_nonsecret_toggles_only(tmp_path: Path):
    """unset-azure-openai removes Azure toggles from JSON store (no secrets there anyway)."""
    runner = CliRunner()

    # First set Azure (creates non-secret toggles)
    set_res = runner.invoke(
        app,
        [
            "set-azure-openai",
            "--openai-endpoint",
            "https://fake-endpoint.openai.azure.com",
            "--openai-api-key",
            "sk-ignored",
            "--deployment-name",
            "fake-deployment",
            "--openai-api-version",
            "2024-01-01",
            "--openai-model-name",
            "gpt-4o-mini",
        ],
    )
    assert set_res.exit_code == 0, set_res.output
    assert (
        _read_hidden_store(tmp_path).get(ModelKeyValues.USE_AZURE_OPENAI.value)
        == "YES"
    )

    # Now unset
    unset_res = runner.invoke(app, ["unset-azure-openai"])
    assert unset_res.exit_code == 0, unset_res.output

    data = _read_hidden_store(tmp_path)
    # Azure toggles should be gone
    for k in [
        ModelKeyValues.USE_AZURE_OPENAI.value,
        ModelKeyValues.AZURE_MODEL_NAME.value,
        ModelKeyValues.AZURE_OPENAI_ENDPOINT.value,
        ModelKeyValues.OPENAI_API_VERSION.value,
        ModelKeyValues.AZURE_DEPLOYMENT_NAME.value,
        ModelKeyValues.AZURE_MODEL_VERSION.value,
    ]:
        assert k not in data
