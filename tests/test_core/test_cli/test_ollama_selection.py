from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import deepeval.metrics.utils.models as model_selection
from deepeval.cli.dotenv_handler import DotenvHandler
from deepeval.cli.main import app as cli_app
from deepeval.config.settings import Settings, reset_settings
from tests.test_core.stubs import make_fake_ollama_module

pytestmark = pytest.mark.enable_dotenv

SAVE = ["--save", "dotenv"]
LOCAL_BASE_URL = "http://localhost:8000/v1"
SET_OLLAMA = ["set-ollama", "--model", "llama3.2"]
SET_OLLAMA_EMBEDDINGS = ["set-ollama-embeddings", "--model", "nomic-embed-text"]
SET_LOCAL_MODEL = [
    "set-local-model",
    "--model",
    "my-vllm-model",
    "--base-url",
    LOCAL_BASE_URL,
]
SET_LOCAL_EMBEDDINGS = [
    "set-local-embeddings",
    "--model",
    "my-embedder",
    "--base-url",
    LOCAL_BASE_URL,
]
SET_AZURE = [
    "set-azure-openai",
    "--model",
    "gpt-4o",
    "--deployment-name",
    "my-deployment",
    "--base-url",
    "https://example.openai.azure.com",
    "--api-version",
    "2024-06-01",
]

LEGACY_OLLAMA_DOTENV = (
    "USE_LOCAL_MODEL=true\n"
    "LOCAL_MODEL_API_KEY=ollama\n"
    "OLLAMA_MODEL_NAME=llama3.2\n"
)
WIZARD_OLLAMA_DOTENV = "USE_OLLAMA_MODEL=true\n" + LEGACY_OLLAMA_DOTENV
LEGACY_OLLAMA_EMBEDDINGS_DOTENV = (
    "USE_LOCAL_EMBEDDINGS=true\n"
    "LOCAL_EMBEDDING_API_KEY=ollama\n"
    "LOCAL_EMBEDDING_MODEL_NAME=nomic-embed-text\n"
)


def _next_process(monkeypatch) -> None:
    for key in Settings.model_fields:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    settings = reset_settings(reload_dotenv=True)
    monkeypatch.setattr(model_selection, "SETTINGS", settings)


def _invoke(monkeypatch, argv: List[str], input: Optional[str] = None):
    _next_process(monkeypatch)
    result = CliRunner().invoke(cli_app, argv, input=input)
    assert result.exit_code == 0, result.output
    return result


def _run(monkeypatch, commands: Iterable[List[str]]) -> None:
    for argv in commands:
        _invoke(monkeypatch, argv)
    _next_process(monkeypatch)


def _write_dotenv(tmp_path: Path, content: str) -> None:
    (tmp_path / ".env.local").write_text(content)


def _secret(name: str) -> Optional[str]:
    value = getattr(model_selection.SETTINGS, name)
    return value.get_secret_value() if value is not None else None


def _llm_choice() -> str:
    if model_selection.should_use_ollama_model():
        return "ollama"
    if model_selection.should_use_local_model():
        return "local"
    if model_selection.should_use_azure_openai():
        return "azure"
    return "default"


def _embedding_choice() -> str:
    if model_selection.should_use_ollama_embedding():
        return "ollama"
    if model_selection.should_use_local_embedding():
        return "local"
    return "default"


@pytest.fixture
def fake_ollama():
    module = make_fake_ollama_module()
    with patch(
        "deepeval.models.llms.ollama_model.require_dependency",
        return_value=module,
    ), patch(
        "deepeval.models.embedding_models.ollama_embedding_model"
        ".require_dependency",
        return_value=module,
    ):
        yield


@pytest.mark.parametrize("save", [[], SAVE], ids=["no-save", "save"])
def test_set_ollama_builds_ollama_in_the_next_process(
    monkeypatch, fake_ollama, save
):
    _run(monkeypatch, [[*SET_OLLAMA, *save]])

    model, _ = model_selection.initialize_model()

    assert type(model).__name__ == "OllamaModel"
    assert model.name == "llama3.2"


def test_set_local_model_after_set_ollama_selects_local(monkeypatch):
    _run(monkeypatch, [SET_OLLAMA])
    assert _llm_choice() == "ollama"

    _run(monkeypatch, [SET_LOCAL_MODEL])

    assert _llm_choice() == "local"


def test_saved_set_local_model_after_set_ollama_builds_local(
    monkeypatch, tmp_path
):
    _run(monkeypatch, [[*SET_OLLAMA, *SAVE]])
    assert _llm_choice() == "ollama"

    _invoke(
        monkeypatch,
        [*SET_LOCAL_MODEL, "--prompt-api-key", *SAVE],
        input="real-key\n",
    )
    _next_process(monkeypatch)
    model, _ = model_selection.initialize_model()

    assert type(model).__name__ == "LocalModel"
    assert model.name == "my-vllm-model"
    assert model.base_url == LOCAL_BASE_URL
    assert _secret("LOCAL_MODEL_API_KEY") == "real-key"
    assert "USE_OLLAMA_MODEL" not in (tmp_path / ".env.local").read_text()


def test_set_azure_after_set_ollama_selects_azure(monkeypatch):
    _run(monkeypatch, [[*SET_OLLAMA, *SAVE], [*SET_AZURE, *SAVE]])

    assert _llm_choice() == "azure"
    assert _secret("LOCAL_MODEL_API_KEY") is None


def test_an_older_set_ollama_after_a_saved_switch_selects_ollama(
    monkeypatch, tmp_path
):
    _run(monkeypatch, [[*SET_OLLAMA, *SAVE], [*SET_AZURE, *SAVE]])
    DotenvHandler(tmp_path / ".env.local").upsert(
        {"USE_LOCAL_MODEL": "true", "LOCAL_MODEL_API_KEY": "ollama"}
    )
    _next_process(monkeypatch)

    assert _llm_choice() == "ollama"


@pytest.mark.parametrize(
    "commands",
    [
        [SET_AZURE, ["unset-azure-openai"]],
        [SET_LOCAL_MODEL, ["unset-ollama"]],
    ],
    ids=["switch-and-unset-azure", "unset-ollama-while-local"],
)
def test_commands_that_never_selected_ollama_keep_a_later_legacy_config(
    monkeypatch, tmp_path, commands
):
    _run(monkeypatch, commands)
    _write_dotenv(tmp_path, LEGACY_OLLAMA_DOTENV)
    _next_process(monkeypatch)

    assert _llm_choice() == "ollama"


@pytest.mark.parametrize("save", [[], SAVE], ids=["no-save", "save"])
def test_set_local_model_overrides_a_legacy_ollama_dotenv(
    monkeypatch, tmp_path, save
):
    _write_dotenv(tmp_path, LEGACY_OLLAMA_DOTENV)
    _run(monkeypatch, [[*SET_LOCAL_MODEL, *save]])

    assert _llm_choice() == "local"


def test_wizard_selecting_ollama_overrides_an_earlier_switch_away(
    monkeypatch, tmp_path, hidden_store_dir
):
    _write_dotenv(tmp_path, LEGACY_OLLAMA_DOTENV)
    _run(monkeypatch, [SET_LOCAL_MODEL])
    stored = json.loads((hidden_store_dir / ".deepeval").read_text())
    assert stored["USE_OLLAMA_MODEL"] == "NO"
    assert _llm_choice() == "local"

    _write_dotenv(tmp_path, WIZARD_OLLAMA_DOTENV)
    _next_process(monkeypatch)

    assert _llm_choice() == "ollama"


@pytest.mark.parametrize(
    "legacy_dotenv", [False, True], ids=["set-ollama", "legacy-dotenv"]
)
def test_unset_ollama_with_save_deselects_ollama(
    monkeypatch, tmp_path, legacy_dotenv
):
    if legacy_dotenv:
        _write_dotenv(tmp_path, LEGACY_OLLAMA_DOTENV)
        _next_process(monkeypatch)
    else:
        _run(monkeypatch, [[*SET_OLLAMA, *SAVE]])
    assert _llm_choice() == "ollama"

    _run(monkeypatch, [["unset-ollama", *SAVE]])

    assert _llm_choice() == "default"
    assert _secret("LOCAL_MODEL_API_KEY") is None
    assert "USE_OLLAMA_MODEL" not in (tmp_path / ".env.local").read_text()


@pytest.mark.parametrize(
    "save, expected_key",
    [([], None), (SAVE, "real-key")],
    ids=["no-save", "save"],
)
def test_unset_ollama_keeps_an_active_local_model_and_its_config(
    monkeypatch, save, expected_key
):
    _run(monkeypatch, [[*SET_OLLAMA, *save]])
    _invoke(
        monkeypatch,
        [*SET_LOCAL_MODEL, "--prompt-api-key", *save],
        input="real-key\n",
    )
    _run(monkeypatch, [["unset-ollama", *save]])

    assert _llm_choice() == "local"
    settings = model_selection.SETTINGS
    assert settings.LOCAL_MODEL_NAME == "my-vllm-model"
    assert str(settings.LOCAL_MODEL_BASE_URL) == LOCAL_BASE_URL
    assert _secret("LOCAL_MODEL_API_KEY") == expected_key


@pytest.mark.parametrize(
    "selected, dotenv, expected",
    [
        ("should_use_ollama_model", "USE_OLLAMA_MODEL=true\n", True),
        (
            "should_use_ollama_model",
            "USE_OLLAMA_MODEL=false\n" + LEGACY_OLLAMA_DOTENV,
            False,
        ),
        (
            "should_use_ollama_model",
            "USE_OLLAMA_MODEL=\n" + LEGACY_OLLAMA_DOTENV,
            False,
        ),
        ("should_use_ollama_model", LEGACY_OLLAMA_DOTENV, True),
        ("should_use_ollama_embedding", "USE_OLLAMA_EMBEDDINGS=true\n", True),
        (
            "should_use_ollama_embedding",
            "USE_OLLAMA_EMBEDDINGS=false\n" + LEGACY_OLLAMA_EMBEDDINGS_DOTENV,
            False,
        ),
        (
            "should_use_ollama_embedding",
            "USE_OLLAMA_EMBEDDINGS=\n" + LEGACY_OLLAMA_EMBEDDINGS_DOTENV,
            False,
        ),
        ("should_use_ollama_embedding", LEGACY_OLLAMA_EMBEDDINGS_DOTENV, True),
    ],
    ids=[
        "llm-true",
        "llm-false-over-marker",
        "llm-empty-over-marker",
        "llm-legacy-marker",
        "embeddings-true",
        "embeddings-false-over-marker",
        "embeddings-empty-over-marker",
        "embeddings-legacy-marker",
    ],
)
def test_a_present_ollama_flag_decides_before_the_legacy_marker(
    monkeypatch, tmp_path, selected, dotenv, expected
):
    _write_dotenv(tmp_path, dotenv)
    _next_process(monkeypatch)

    assert getattr(model_selection, selected)() is expected


@pytest.mark.parametrize(
    "set_command, dotenv, selected",
    [
        (SET_OLLAMA, "USE_OLLAMA_MODEL=false\n", "should_use_ollama_model"),
        (
            SET_OLLAMA_EMBEDDINGS,
            "USE_OLLAMA_EMBEDDINGS=false\n",
            "should_use_ollama_embedding",
        ),
    ],
    ids=["llm", "embeddings"],
)
def test_a_dotenv_flag_outranks_the_keystore(
    monkeypatch, tmp_path, set_command, dotenv, selected
):
    _run(monkeypatch, [set_command])
    assert getattr(model_selection, selected)()

    _write_dotenv(tmp_path, dotenv)
    _next_process(monkeypatch)

    assert not getattr(model_selection, selected)()


@pytest.mark.parametrize("save", [[], SAVE], ids=["no-save", "save"])
def test_set_ollama_embeddings_builds_ollama_in_the_next_process(
    monkeypatch, fake_ollama, save
):
    _run(monkeypatch, [[*SET_OLLAMA_EMBEDDINGS, *save]])

    model = model_selection.initialize_embedding_model()

    assert type(model).__name__ == "OllamaEmbeddingModel"
    assert model.name == "nomic-embed-text"


def test_set_local_embeddings_after_ollama_embeddings_selects_local(
    monkeypatch,
):
    _run(monkeypatch, [[*SET_OLLAMA_EMBEDDINGS, *SAVE]])
    assert _embedding_choice() == "ollama"

    _run(monkeypatch, [[*SET_LOCAL_EMBEDDINGS, *SAVE]])

    assert _embedding_choice() == "local"


@pytest.mark.parametrize(
    "legacy_dotenv",
    [False, True],
    ids=["set-ollama-embeddings", "legacy-dotenv"],
)
def test_unset_ollama_embeddings_with_save_deselects_ollama(
    monkeypatch, tmp_path, legacy_dotenv
):
    if legacy_dotenv:
        _write_dotenv(tmp_path, LEGACY_OLLAMA_EMBEDDINGS_DOTENV)
        _next_process(monkeypatch)
    else:
        _run(monkeypatch, [[*SET_OLLAMA_EMBEDDINGS, *SAVE]])
    assert _embedding_choice() == "ollama"

    _run(monkeypatch, [["unset-ollama-embeddings", *SAVE]])

    assert _embedding_choice() == "default"
    assert _secret("LOCAL_EMBEDDING_API_KEY") is None
    content = (tmp_path / ".env.local").read_text()
    assert "USE_OLLAMA_EMBEDDINGS" not in content
    assert "USE_LOCAL_EMBEDDINGS" not in content
    assert "LOCAL_EMBEDDING_MODEL_NAME" not in content


@pytest.mark.parametrize("save", [[], SAVE], ids=["no-save", "save"])
def test_unset_ollama_embeddings_keeps_active_local_embeddings(
    monkeypatch, save
):
    _run(
        monkeypatch,
        [
            [*SET_OLLAMA_EMBEDDINGS, *save],
            [*SET_LOCAL_EMBEDDINGS, *save],
            ["unset-ollama-embeddings", *save],
        ],
    )

    assert _embedding_choice() == "local"
    settings = model_selection.SETTINGS
    assert settings.LOCAL_EMBEDDING_MODEL_NAME == "my-embedder"
    assert str(settings.LOCAL_EMBEDDING_BASE_URL) == LOCAL_BASE_URL


def test_legacy_keystore_embedding_selector_selects_ollama(
    monkeypatch, hidden_store_dir: Path
):
    (hidden_store_dir / ".deepeval").write_text(
        '{"LOCAL_EMBEDDING_API_KEY": "ollama"}'
    )
    _next_process(monkeypatch)

    assert model_selection.should_use_ollama_embedding()


def test_llm_and_embedding_ollama_selections_are_independent(monkeypatch):
    _run(monkeypatch, [SET_OLLAMA_EMBEDDINGS, SET_LOCAL_MODEL])
    assert _llm_choice() == "local"
    assert _embedding_choice() == "ollama"

    _run(monkeypatch, [SET_OLLAMA, SET_LOCAL_EMBEDDINGS])
    assert _llm_choice() == "ollama"
    assert _embedding_choice() == "local"
