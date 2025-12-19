from __future__ import annotations

import json
import re
import pytest
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple
from typer.testing import CliRunner
from dataclasses import dataclass

from deepeval.cli.main import app as cli_app
from deepeval.cli.utils import USE_EMBED_KEYS, USE_LLM_KEYS
from deepeval.config.settings import Settings, reset_settings  # noqa: E402
from deepeval.config.utils import parse_bool


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Box drawing block used by rich panels (┌─┐│└─┘ etc.)
_BOX_RE = re.compile(r"[\u2500-\u257F]")


def _normalize_cli_output(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    text = _BOX_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_hidden_store_json(hidden_store_dir: Path) -> Dict[str, object]:
    """
    The cli_app writes the legacy key/value store to: <cwd>/.deepeval/.deepeval
    """
    store_file = hidden_store_dir / ".deepeval"
    if not store_file.exists():
        return {}

    raw_text = store_file.read_text(encoding="utf-8").strip()
    if not raw_text:
        return {}

    return json.loads(raw_text)


def _assert_no_dupes(env_path: Path, keys: Iterable[str]) -> None:
    for key in keys:
        occurrences = _count_key_occurrences(env_path, key)
        assert occurrences <= 1, f"Key {key} duplicated in {env_path}"


def _assert_use_flags_exclusive_env(
    env_path: Path,
    selected_key: str,
    all_use_keys: List[str],
) -> None:
    env_vars = _read_dotenv(env_path)

    assert parse_bool(
        env_vars.get(selected_key)
    ), f"Expected {selected_key} to be truthy in {env_path}"

    for use_key in all_use_keys:
        if use_key == selected_key:
            continue
        # It's OK if a "false" flag isn't written at all; if it is present, it must be falsey.
        if use_key in env_vars:
            assert not parse_bool(
                env_vars.get(use_key)
            ), f"Expected {use_key} to be falsey in {env_path}"


def _assert_use_flags_exclusive_store(
    store: Mapping[str, object],
    selected_key: str,
    all_use_keys: List[str],
) -> None:
    for use_key in all_use_keys:
        stored_value = store.get(use_key)
        if use_key == selected_key:
            assert (
                stored_value == "YES"
            ), f"Expected {use_key}=YES in .deepeval store, got {stored_value!r}"
        else:
            assert (
                stored_value == "NO"
            ), f"Expected {use_key}=NO in .deepeval store, got {stored_value!r}"


def _unquote_dotenv_value(value: str) -> str:
    stripped = value.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in ('"', "'")
    ):
        return stripped[1:-1]
    return stripped


def _read_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    env_vars: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        env_vars[key.strip()] = _unquote_dotenv_value(raw_value)

    return env_vars


def _count_key_occurrences(path: Path, key: str) -> int:
    if not path.exists():
        return 0

    prefix = f"{key}="
    return sum(
        1
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if raw_line.startswith(prefix)
    )


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _invoke_ok(runner: CliRunner, argv: list[str]) -> str:
    result = runner.invoke(cli_app, argv, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def test_settings_set_coerces_and_persists_dotenv(
    runner: CliRunner, env_path: Path, settings: Settings
) -> None:
    argv = [
        "settings",
        "-u",
        "log-level=error",
        "-u",
        "temperature=0.92",
        "--save",
        f"dotenv:{env_path}",
    ]

    _invoke_ok(runner, argv)

    env = _read_dotenv(env_path)
    # LOG_LEVEL is validated/coerced by Settings
    assert env.get("LOG_LEVEL") == "40"
    assert env.get("TEMPERATURE") == "0.92"

    # In-memory settings singleton should reflect the update.
    assert settings.LOG_LEVEL == 40
    assert settings.TEMPERATURE == pytest.approx(0.92)

    # Running again should be a no-op and should not duplicate keys in the dotenv file.
    out2 = _invoke_ok(runner, argv)
    assert "No changes to save" in out2
    assert _count_key_occurrences(env_path, "LOG_LEVEL") == 1
    assert _count_key_occurrences(env_path, "TEMPERATURE") == 1


def test_settings_unset_removes_key_from_dotenv(
    runner: CliRunner, env_path: Path, settings: Settings
) -> None:
    _invoke_ok(
        runner,
        [
            "settings",
            "-u",
            "temperature=0.5",
            "--save",
            f"dotenv:{env_path}",
        ],
    )
    assert _read_dotenv(env_path).get("TEMPERATURE") == "0.5"

    _invoke_ok(
        runner,
        [
            "settings",
            "-U",
            "temperature",
            "--save",
            f"dotenv:{env_path}",
        ],
    )

    settings = reset_settings(reload_dotenv=False)
    env = _read_dotenv(env_path)
    assert "TEMPERATURE" not in env
    assert settings.TEMPERATURE is None


def test_settings_list_filters_and_masks_secrets(
    runner: CliRunner,
    env_path: Path,
    settings: Settings,
    hidden_store_dir: Path,
) -> None:
    # Set a secret value via the Settings command.
    _invoke_ok(
        runner,
        [
            "settings",
            "-u",
            "anthropic-api-key=sk-test",
            "--save",
            f"dotenv:{env_path}",
        ],
    )

    env = _read_dotenv(env_path)
    assert env.get("ANTHROPIC_API_KEY") == "sk-test"

    # Secrets should never be persisted into the legacy JSON store.
    store_path = hidden_store_dir / ".deepeval"
    store = _read_hidden_store_json(store_path)
    assert "ANTHROPIC_API_KEY" not in store

    # The --list output should mask the secret (and not echo the raw value).
    out = _invoke_ok(runner, ["settings", "-l", "anthropic"])
    assert "ANTHROPIC_API_KEY" in out
    assert "********" in out
    assert "sk-test" not in out


def test_set_debug_quiet_suppresses_output_and_updates_dotenv(
    runner: CliRunner, env_path: Path
) -> None:
    result = runner.invoke(
        cli_app,
        [
            "set-debug",
            "--log-level",
            "DEBUG",
            "--save",
            f"dotenv:{env_path}",
            "--quiet",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert result.output.strip() == ""

    env = _read_dotenv(env_path)
    # DEBUG should be coerced to the numeric level (10).
    assert env.get("LOG_LEVEL") == "10"


@dataclass(frozen=True)
class _ProviderCase:
    set_cmd: str
    unset_cmd: str
    use_key: str
    set_flags: Tuple[str, ...]
    expected_env: Dict[str, str]
    expected_store: Dict[str, str]


LLM_PROVIDER_CASES: List[_ProviderCase] = [
    _ProviderCase(
        set_cmd="set-openai",
        unset_cmd="unset-openai",
        use_key="USE_OPENAI_MODEL",
        set_flags=("--model", "gpt-4o-mini"),
        expected_env={
            "OPENAI_MODEL_NAME": "gpt-4o-mini",
        },
        expected_store={
            "OPENAI_MODEL_NAME": "gpt-4o-mini",
        },
    ),
    _ProviderCase(
        set_cmd="set-azure-openai",
        unset_cmd="unset-azure-openai",
        use_key="USE_AZURE_OPENAI",
        set_flags=(
            "--model",
            "gpt-4.1",
            "--deployment-name",
            "dep1",
            "--base-url",
            "https://example.openai.azure.com/",
            "--api-version",
            "2024-06-01",
        ),
        expected_env={
            "AZURE_MODEL_NAME": "gpt-4.1",
            "AZURE_DEPLOYMENT_NAME": "dep1",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "OPENAI_API_VERSION": "2024-06-01",
        },
        expected_store={
            "AZURE_MODEL_NAME": "gpt-4.1",
            "AZURE_DEPLOYMENT_NAME": "dep1",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "OPENAI_API_VERSION": "2024-06-01",
        },
    ),
    _ProviderCase(
        set_cmd="set-anthropic",
        unset_cmd="unset-anthropic",
        use_key="USE_ANTHROPIC_MODEL",
        set_flags=("--model", "claude-3-5-haiku-latest"),
        expected_env={"ANTHROPIC_MODEL_NAME": "claude-3-5-haiku-latest"},
        expected_store={"ANTHROPIC_MODEL_NAME": "claude-3-5-haiku-latest"},
    ),
    _ProviderCase(
        set_cmd="set-bedrock",
        unset_cmd="unset-bedrock",
        use_key="USE_AWS_BEDROCK_MODEL",
        set_flags=("--model", "anthropic.claude-v2", "--region", "us-east-1"),
        expected_env={
            "AWS_BEDROCK_MODEL_NAME": "anthropic.claude-v2",
            "AWS_BEDROCK_REGION": "us-east-1",
        },
        expected_store={
            "AWS_BEDROCK_MODEL_NAME": "anthropic.claude-v2",
            "AWS_BEDROCK_REGION": "us-east-1",
        },
    ),
    _ProviderCase(
        set_cmd="set-ollama",
        unset_cmd="unset-ollama",
        use_key="USE_LOCAL_MODEL",
        set_flags=(
            "--model",
            "llama3",
            "--base-url",
            "http://localhost:11434/",
        ),
        expected_env={
            "OLLAMA_MODEL_NAME": "llama3",
            "LOCAL_MODEL_BASE_URL": "http://localhost:11434/",
        },
        expected_store={
            "OLLAMA_MODEL_NAME": "llama3",
            "LOCAL_MODEL_BASE_URL": "http://localhost:11434/",
        },
    ),
    _ProviderCase(
        set_cmd="set-local-model",
        unset_cmd="unset-local-model",
        use_key="USE_LOCAL_MODEL",
        set_flags=(
            "--model",
            "my-local",
            "--base-url",
            "http://localhost:8000/",
            "--format",
            "openai",
        ),
        expected_env={
            "LOCAL_MODEL_NAME": "my-local",
            "LOCAL_MODEL_BASE_URL": "http://localhost:8000/",
            "LOCAL_MODEL_FORMAT": "openai",
        },
        expected_store={
            "LOCAL_MODEL_NAME": "my-local",
            "LOCAL_MODEL_BASE_URL": "http://localhost:8000/",
            "LOCAL_MODEL_FORMAT": "openai",
        },
    ),
    _ProviderCase(
        set_cmd="set-grok",
        unset_cmd="unset-grok",
        use_key="USE_GROK_MODEL",
        set_flags=("--model", "grok-2"),
        expected_env={"GROK_MODEL_NAME": "grok-2"},
        expected_store={"GROK_MODEL_NAME": "grok-2"},
    ),
    _ProviderCase(
        set_cmd="set-moonshot",
        unset_cmd="unset-moonshot",
        use_key="USE_MOONSHOT_MODEL",
        set_flags=("--model", "moonshot-v1"),
        expected_env={"MOONSHOT_MODEL_NAME": "moonshot-v1"},
        expected_store={"MOONSHOT_MODEL_NAME": "moonshot-v1"},
    ),
    _ProviderCase(
        set_cmd="set-deepseek",
        unset_cmd="unset-deepseek",
        use_key="USE_DEEPSEEK_MODEL",
        set_flags=("--model", "deepseek-chat"),
        expected_env={"DEEPSEEK_MODEL_NAME": "deepseek-chat"},
        expected_store={"DEEPSEEK_MODEL_NAME": "deepseek-chat"},
    ),
    _ProviderCase(
        set_cmd="set-gemini",
        unset_cmd="unset-gemini",
        use_key="USE_GEMINI_MODEL",
        set_flags=(
            "--model",
            "gemini-1.5-pro",
            "--project",
            "my-proj",
            "--location",
            "us-central1",
        ),
        expected_env={
            "GEMINI_MODEL_NAME": "gemini-1.5-pro",
            "GOOGLE_CLOUD_PROJECT": "my-proj",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
        expected_store={
            "GEMINI_MODEL_NAME": "gemini-1.5-pro",
            "GOOGLE_CLOUD_PROJECT": "my-proj",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
    ),
    _ProviderCase(
        set_cmd="set-litellm",
        unset_cmd="unset-litellm",
        use_key="USE_LITELLM",
        set_flags=(
            "--model",
            "gpt-4.1",
            "--base-url",
            "http://localhost:4000/",
            "--proxy-base-url",
            "http://localhost:5000/",
        ),
        expected_env={
            "LITELLM_MODEL_NAME": "gpt-4.1",
            "LITELLM_API_BASE": "http://localhost:4000/",
            "LITELLM_PROXY_API_BASE": "http://localhost:5000/",
        },
        expected_store={
            "LITELLM_MODEL_NAME": "gpt-4.1",
            "LITELLM_API_BASE": "http://localhost:4000/",
            "LITELLM_PROXY_API_BASE": "http://localhost:5000/",
        },
    ),
    _ProviderCase(
        set_cmd="set-portkey",
        unset_cmd="unset-portkey",
        use_key="USE_PORTKEY_MODEL",
        set_flags=(
            "--model",
            "gpt-4.1",
            "--base-url",
            "http://localhost:8787/",
            "--provider",
            "openai",
        ),
        expected_env={
            "PORTKEY_MODEL_NAME": "gpt-4.1",
            "PORTKEY_BASE_URL": "http://localhost:8787/",
            "PORTKEY_PROVIDER_NAME": "openai",
        },
        expected_store={
            "PORTKEY_MODEL_NAME": "gpt-4.1",
            "PORTKEY_BASE_URL": "http://localhost:8787/",
            "PORTKEY_PROVIDER_NAME": "openai",
        },
    ),
]


@pytest.mark.parametrize("case", LLM_PROVIDER_CASES, ids=lambda c: c.set_cmd)
def test_set_unset_llm_provider_roundtrip(
    case: _ProviderCase,
    hidden_store_dir: Path,
    env_path: Path,
) -> None:
    runner = CliRunner()
    save = f"dotenv:{env_path}"

    # Force a real transition so we can assert persistence deterministically.
    # we don't persist what hasn't changed, therefore USE_* of the default provider.
    # won't persist unless we unset first.
    result = runner.invoke(cli_app, [case.unset_cmd, "--save", save])
    assert result.exit_code == 0, result.output

    # --- set ---
    result = runner.invoke(
        cli_app, [case.set_cmd, *case.set_flags, "--save", save]
    )
    assert result.exit_code == 0, result.output

    store = _read_hidden_store_json(hidden_store_dir)
    env = _read_dotenv(env_path)

    _assert_no_dupes(env_path, list(case.expected_env.keys()) + USE_LLM_KEYS)
    for k, v in case.expected_env.items():
        assert env.get(k) == v, f"env={env}\nstore={store}"
    _assert_use_flags_exclusive_env(env_path, case.use_key, USE_LLM_KEYS)

    _assert_use_flags_exclusive_store(store, case.use_key, USE_LLM_KEYS)
    for k, v in case.expected_store.items():
        assert (
            store.get(k) == v
        ), f"Expected {k}={v!r} in .deepeval store, got {store.get(k)!r}"

    # unset
    result2 = runner.invoke(cli_app, [case.unset_cmd, "--save", save])
    assert result2.exit_code == 0, result2.output

    env2 = _read_dotenv(env_path)
    # provider keys cleared
    for k in case.expected_env.keys():
        assert k not in env2
    # use flag disabled (either removed or set falsey)
    assert not parse_bool(env2.get(case.use_key))
    # and nothing remains enabled in the store
    store2 = _read_hidden_store_json(hidden_store_dir)
    for k in case.expected_store.keys():
        assert k not in store2

    assert store2.get(case.use_key) in {
        None,
        "NO",
    }, f"Expected {case.use_key} cleared in store after unset"
    assert "YES" not in [
        store2.get(k) for k in USE_LLM_KEYS
    ], "Expected no LLM USE_* key to remain YES after unset"


EMBED_PROVIDER_CASES: List[_ProviderCase] = [
    _ProviderCase(
        set_cmd="set-azure-openai-embedding",
        unset_cmd="unset-azure-openai-embedding",
        use_key="USE_AZURE_OPENAI_EMBEDDING",
        set_flags=(
            "--model",
            "text-embedding-3-large",
            "--deployment-name",
            "embed-dep",
        ),
        expected_env={
            "AZURE_EMBEDDING_MODEL_NAME": "text-embedding-3-large",
            "AZURE_EMBEDDING_DEPLOYMENT_NAME": "embed-dep",
        },
        expected_store={
            "AZURE_EMBEDDING_MODEL_NAME": "text-embedding-3-large",
            "AZURE_EMBEDDING_DEPLOYMENT_NAME": "embed-dep",
        },
    ),
    _ProviderCase(
        set_cmd="set-local-embeddings",
        unset_cmd="unset-local-embeddings",
        use_key="USE_LOCAL_EMBEDDINGS",
        set_flags=(
            "--model",
            "nomic-embed-text",
            "--base-url",
            "http://localhost:8000/",
        ),
        expected_env={
            "LOCAL_EMBEDDING_MODEL_NAME": "nomic-embed-text",
            "LOCAL_EMBEDDING_BASE_URL": "http://localhost:8000/",
        },
        expected_store={
            "LOCAL_EMBEDDING_MODEL_NAME": "nomic-embed-text",
            "LOCAL_EMBEDDING_BASE_URL": "http://localhost:8000/",
        },
    ),
    _ProviderCase(
        set_cmd="set-ollama-embeddings",
        unset_cmd="unset-ollama-embeddings",
        use_key="USE_LOCAL_EMBEDDINGS",
        set_flags=(
            "--model",
            "nomic-embed-text",
            "--base-url",
            "http://localhost:11434/",
        ),
        expected_env={
            "LOCAL_EMBEDDING_MODEL_NAME": "nomic-embed-text",
            "LOCAL_EMBEDDING_BASE_URL": "http://localhost:11434/",
        },
        expected_store={
            "LOCAL_EMBEDDING_MODEL_NAME": "nomic-embed-text",
            "LOCAL_EMBEDDING_BASE_URL": "http://localhost:11434/",
        },
    ),
]


@pytest.mark.parametrize("case", EMBED_PROVIDER_CASES, ids=lambda c: c.set_cmd)
def test_set_unset_embedding_provider_roundtrip(
    case: _ProviderCase,
    hidden_store_dir: Path,
    env_path: Path,
) -> None:
    runner = CliRunner()
    save = f"dotenv:{env_path}"

    # unset first to deal with default provider
    runner.invoke(cli_app, [case.unset_cmd, "--save", save])
    # set
    result = runner.invoke(
        cli_app, [case.set_cmd, *case.set_flags, "--save", save]
    )
    assert result.exit_code == 0, result.output

    env = _read_dotenv(env_path)
    _assert_no_dupes(env_path, list(case.expected_env.keys()) + USE_EMBED_KEYS)
    for k, v in case.expected_env.items():
        assert env.get(k) == v
    _assert_use_flags_exclusive_env(env_path, case.use_key, USE_EMBED_KEYS)

    store = _read_hidden_store_json(hidden_store_dir)
    _assert_use_flags_exclusive_store(store, case.use_key, USE_EMBED_KEYS)
    for k, v in case.expected_store.items():
        assert (
            store.get(k) == v
        ), f"Expected {k}={v!r} in .deepeval store, got {store.get(k)!r}"

    # unset
    result2 = runner.invoke(cli_app, [case.unset_cmd, "--save", save])
    assert result2.exit_code == 0, result2.output

    env2 = _read_dotenv(env_path)
    for k in case.expected_env.keys():
        assert k not in env2
    assert not parse_bool(
        env2.get(case.use_key)
    ), f"Expected {case.use_key} to be disabled after unset"

    store2 = _read_hidden_store_json(hidden_store_dir)
    for k in case.expected_store.keys():
        assert k not in store2
    assert store2.get(case.use_key) in {
        None,
        "NO",
    }, f"Expected {case.use_key} cleared in store after unset"
    assert "YES" not in [
        store2.get(k) for k in USE_EMBED_KEYS
    ], "Expected no embedding USE_* key to remain YES after unset"


def test_set_unset_gemini_service_account_file_roundtrip_dotenv_only(
    tmp_path,
    hidden_store_dir,
    env_path,
) -> None:
    runner = CliRunner()
    save = f"dotenv:{env_path}"

    # Create a real JSON file (with whitespace) so we can verify normalization.
    sa_obj = {
        "type": "service_account",
        "project_id": "my-proj",
        "private_key_id": "abc123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
        "client_email": "x@y.z",
    }
    sa_file = tmp_path / "sa.json"
    sa_file.write_text(json.dumps(sa_obj, indent=2), encoding="utf-8")
    expected_sa = json.dumps(
        sa_obj, separators=(",", ":")
    )  # matches loader behavior

    # Force a real transition
    result = runner.invoke(cli_app, ["unset-gemini", "--save", save])
    assert result.exit_code == 0, result.output

    # set (Vertex path)
    result = runner.invoke(
        cli_app,
        [
            "set-gemini",
            "--model",
            "gemini-1.5-pro",
            "--project",
            "my-proj",
            "--location",
            "us-central1",
            "--service-account-file",
            str(sa_file),
            "--save",
            save,
        ],
    )
    assert result.exit_code == 0, result.output

    env = _read_dotenv(env_path)

    # Service account key should be persisted to dotenv as a single line.
    assert env.get("GOOGLE_SERVICE_ACCOUNT_KEY") == expected_sa

    # Because project/location/service-account-file set and no api_key, Vertex mode should be enabled.
    assert parse_bool(env.get("GOOGLE_GENAI_USE_VERTEXAI")) is True

    # And secrets should not land in the legacy JSON store.
    store = _read_hidden_store_json(hidden_store_dir)
    assert "GOOGLE_SERVICE_ACCOUNT_KEY" not in store

    # unset with --clear-secrets clears it from dotenv
    result = runner.invoke(
        cli_app, ["unset-gemini", "--clear-secrets", "--save", save]
    )
    assert result.exit_code == 0, result.output
    env2 = _read_dotenv(env_path)
    assert "GOOGLE_SERVICE_ACCOUNT_KEY" not in env2


def test_set_gemini_prompt_api_key_persists_to_dotenv_not_json(
    hidden_store_dir,
    env_path,
) -> None:
    runner = CliRunner()
    save = f"dotenv:{env_path}"

    result = runner.invoke(cli_app, ["unset-gemini", "--save", save])
    assert result.exit_code == 0, result.output

    # Typer prompt can be satisfied via CliRunner input
    result = runner.invoke(
        cli_app,
        ["set-gemini", "--prompt-api-key", "--save", save],
        input="test-google-api-key\n",
    )
    assert result.exit_code == 0, result.output

    env = _read_dotenv(env_path)
    assert env.get("GOOGLE_API_KEY") == "test-google-api-key"

    # prompt_api_key path explicitly sets Vertex mode false.
    assert parse_bool(env.get("GOOGLE_GENAI_USE_VERTEXAI")) is False

    store = _read_hidden_store_json(hidden_store_dir)
    assert "GOOGLE_API_KEY" not in store


def test_set_gemini_service_account_file_validation_errors(
    tmp_path, env_path
) -> None:
    runner = CliRunner()
    save = f"dotenv:{env_path}"

    # empty file
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    r = runner.invoke(
        cli_app,
        ["set-gemini", "--service-account-file", str(empty), "--save", save],
    )
    assert r.exit_code != 0
    assert "Service account file is empty" in r.output

    # invalid JSON
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    r = runner.invoke(
        cli_app,
        ["set-gemini", "--service-account-file", str(bad), "--save", save],
    )
    assert r.exit_code != 0
    assert "does not contain valid JSON" in _normalize_cli_output(r.output)


def test_settings_set_writes_to_dotenv_even_if_value_already_in_json_store(
    runner: CliRunner,
    env_path: Path,
    hidden_store_dir: Path,
) -> None:
    # Seed the legacy JSON store with a setting not written to dotenv
    store_path = hidden_store_dir / ".deepeval"
    store_path.write_text(json.dumps({"TEMPERATURE": "0.5"}), encoding="utf-8")

    # Settings is a singleton and is already created by autouse fixtures,
    # so we need to rebuild it to pick up the JSON store value we just wrote.
    settings = reset_settings(reload_dotenv=False)
    assert settings.TEMPERATURE == pytest.approx(0.5)

    # Sanity: dotenv is still empty
    assert _read_dotenv(env_path).get("TEMPERATURE") is None

    # Use the CLI to "set" the same setting, but request persistence to dotenv
    _invoke_ok(
        runner,
        [
            "settings",
            "-u",
            "temperature=0.5",
            "--save",
            f"dotenv:{env_path}",
        ],
    )

    # Assert the setting was persisted to dotenv
    env = _read_dotenv(env_path)
    assert env.get("TEMPERATURE") == "0.5"
