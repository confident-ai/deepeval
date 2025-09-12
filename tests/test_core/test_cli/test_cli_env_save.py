from pathlib import Path

import pytest

from deepeval.key_handler import (
    KeyValues,
    EmbeddingKeyValues,
    ModelKeyValues,
)

from .helpers import (
    build_args,
    invoke_ok,
    invoke_with_error,
    read_dotenv_as_dict,
    assert_env_contains,
    assert_env_lacks,
    assert_no_dupes,
    assert_deepeval_json_contains,
    assert_deepeval_json_lacks,
    assert_deepeval_json_model_switched_to,
    assert_model_switched_to,
)

from deepeval.cli.main import app


pytestmark = pytest.mark.skip(
    reason="Temporarily disabled while refactoring settings persistence"
)


def _read_dotenv_as_dict(path: Path) -> dict:
    """Minimal .env parser: KEY=VALUE lines only; ignores blanks/comments."""
    data = {}
    if not path.exists():
        return data

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def _count_key_occurrences(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text().splitlines()
        if line.startswith(f"{key}=")
    )


def test_dotenv_handler_basic_upsert_unset(env_path: Path):
    """Directly test the DotenvHandler upsert/unset helpers (no CLI)."""
    from deepeval.cli.dotenv_handler import DotenvHandler

    h = DotenvHandler(str(env_path))

    # upsert writes new keys
    h.upsert({"FOO": "1", "BAR": "x"})
    text = env_path.read_text()
    assert "FOO=1" in text and "BAR=x" in text

    # upsert updates existing keys (no duplicates)
    h.upsert({"FOO": "2"})
    text = env_path.read_text()
    assert "FOO=2" in text
    assert _count_key_occurrences(env_path, "FOO") == 1

    # unset removes selected keys only
    h.unset(["BAR"])
    data = _read_dotenv_as_dict(env_path)
    assert "BAR" not in data
    assert data["FOO"] == "2"


def test_openai_save_and_unset(env_path: Path):
    """Verify: set-openai --save writes keys; running again upserts; unset removes only those keys."""

    # Setup
    argv = build_args(
        "set-openai",
        options={
            "--model": "gpt-4o-mini",
            "--cost_per_input_token": 0.15,
            "--cost_per_output_token": 0.20,
        },
        save_path=env_path,
    )

    # Run command once
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.15",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        }
    )

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.15",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_OPENAI_MODEL)

    # Confirm save updates things that changed, but does not duplicate or drop variables
    argv = build_args(
        "set-openai",
        options={
            "--model": "gpt-4o-mini",
            "--cost_per_input_token": 0.18,
            "--cost_per_output_token": 0.20,
        },
        save_path=env_path,
    )

    invoke_ok(app, argv)

    assert_deepeval_json_contains(
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.18",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        }
    )

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.18",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_OPENAI_MODEL)

    assert_no_dupes(
        env_path,
        [
            ModelKeyValues.OPENAI_MODEL_NAME,
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
            ModelKeyValues.USE_OPENAI_MODEL,
        ],
    )

    # keep unrelated, unset only OpenAI keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-openai", save_path=env_path)
    invoke_ok(app, argv)
    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.OPENAI_MODEL_NAME,
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
            ModelKeyValues.USE_OPENAI_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_openai_omits_optional_costs_when_absent(env_path: Path):
    """Calling set-openai without cost overrides must not write 'None' into dotenv or JSON."""
    argv = build_args(
        "set-openai",
        options={"--model": "gpt-4o-mini"},
        save_path=env_path,  # imply --save dotenv:env_path
    )
    invoke_ok(app, argv)

    # JSON: model present, costs absent
    assert_deepeval_json_contains(
        {ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini"}
    )
    assert_deepeval_json_lacks(
        [
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
        ]
    )

    # dotenv: model present, costs absent (not "None")
    assert_env_contains(
        env_path,
        {ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini"},
    )
    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
        ],
    )


def test_set_openai_without_save_updates_json_only(tmp_path: Path):

    # Setup
    argv = build_args(
        "set-openai",
        options={
            "--model": "gpt-4o-mini",
            "--cost_per_input_token": 0.15,
            "--cost_per_output_token": 0.20,
        },
    )

    # Run command
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.15",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        }
    )

    assert_deepeval_json_model_switched_to(ModelKeyValues.USE_OPENAI_MODEL)

    default_env_path = tmp_path / ".env.local"
    assert_env_lacks(
        default_env_path,
        [
            ModelKeyValues.OPENAI_MODEL_NAME,
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
            ModelKeyValues.USE_OPENAI_MODEL,
        ],
    )

    # no dotenv file should be created implicitly
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".env.local").exists()


def test_set_openai_without_save_path_stores_in_default_path(tmp_path: Path):

    # Setup
    argv = build_args(
        "set-openai",
        options={
            "--model": "gpt-4o-mini",
            "--cost_per_input_token": 0.15,
            "--cost_per_output_token": 0.20,
        },
        save_kind="dotenv",
    )

    # Run command
    invoke_ok(app, argv)

    # .env.local is created as default. No other dotenv file should be
    # created implicitly
    assert not (tmp_path / ".env").exists()
    assert (tmp_path / ".env.local").exists()

    # Assert that variables persisted to default path
    default_env_path = tmp_path / ".env.local"
    assert_env_contains(
        default_env_path,
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.15",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        },
    )

    assert_model_switched_to(default_env_path, ModelKeyValues.USE_OPENAI_MODEL)


def test_set_openai_with_invalid_save_kind_only_persists_to_json(
    tmp_path: Path,
):

    # Setup
    argv = build_args(
        "set-openai",
        options={
            "--model": "gpt-4o-mini",
            "--cost_per_input_token": 0.15,
            "--cost_per_output_token": 0.20,
        },
        save_kind="foo",
    )

    # Run command
    invoke_ok(app, argv)

    # Assert that variables still persisted to json store as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.OPENAI_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: "0.15",
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: "0.2",
        }
    )

    assert_deepeval_json_model_switched_to(ModelKeyValues.USE_OPENAI_MODEL)

    # Assert that variables did not persisted to default path
    default_env_path = tmp_path / ".env.local"
    assert_env_lacks(
        default_env_path,
        [
            ModelKeyValues.OPENAI_MODEL_NAME,
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
            ModelKeyValues.USE_OPENAI_MODEL,
        ],
    )

    # no dotenv file should be created for invalid kind
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".env.local").exists()


def test_azure_save_and_unset_roundtrip(env_path: Path):
    """Verify: set-azure-openai --save writes keys; running again upserts; unset removes only those keys."""

    # Setup
    argv = build_args(
        "set-azure-openai",
        options={
            "--openai-endpoint": "https://fake-endpoint.openai.azure.com/",
            "--openai-api-key": "sk-test-123",
            "--deployment-name": "fake-deployment",
            "--openai-api-version": "2024-01-01",
            "--openai-model-name": "gpt-4o-mini",
            "--model-version": "1",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)
    assert_deepeval_json_contains(
        {
            ModelKeyValues.AZURE_OPENAI_ENDPOINT: "https://fake-endpoint.openai.azure.com/",
            ModelKeyValues.OPENAI_API_VERSION: "2024-01-01",
            ModelKeyValues.AZURE_DEPLOYMENT_NAME: "fake-deployment",
            ModelKeyValues.AZURE_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.AZURE_MODEL_VERSION: "1",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.AZURE_OPENAI_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.AZURE_OPENAI_API_KEY: "sk-test-123",
            ModelKeyValues.AZURE_OPENAI_ENDPOINT: "https://fake-endpoint.openai.azure.com/",
            ModelKeyValues.OPENAI_API_VERSION: "2024-01-01",
            ModelKeyValues.AZURE_DEPLOYMENT_NAME: "fake-deployment",
            ModelKeyValues.AZURE_MODEL_NAME: "gpt-4o-mini",
            ModelKeyValues.AZURE_MODEL_VERSION: "1",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_AZURE_OPENAI)

    # upsert without dupes
    argv = build_args(
        "set-azure-openai",
        options={
            "--openai-endpoint": "https://new-endpoint.openai.azure.com/",
            "--openai-api-key": "sk-updated",
            "--deployment-name": "fake-deployment",
            "--openai-api-version": "2024-01-01",
            "--openai-model-name": "gpt-4o-mini",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)
    assert_env_contains(
        env_path,
        {
            ModelKeyValues.AZURE_OPENAI_API_KEY: "sk-updated",
            ModelKeyValues.AZURE_OPENAI_ENDPOINT: "https://new-endpoint.openai.azure.com/",
        },
    )
    assert_no_dupes(
        env_path,
        [
            ModelKeyValues.AZURE_OPENAI_API_KEY,
            ModelKeyValues.AZURE_OPENAI_ENDPOINT,
            ModelKeyValues.OPENAI_API_VERSION,
            ModelKeyValues.AZURE_DEPLOYMENT_NAME,
            ModelKeyValues.AZURE_MODEL_NAME,
        ],
    )

    # keep unrelated, unset only Azure keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-azure-openai", save_path=env_path)
    invoke_ok(app, argv)
    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.AZURE_OPENAI_API_KEY,
            ModelKeyValues.AZURE_OPENAI_ENDPOINT,
            ModelKeyValues.OPENAI_API_VERSION,
            ModelKeyValues.AZURE_DEPLOYMENT_NAME,
            ModelKeyValues.AZURE_MODEL_NAME,
            ModelKeyValues.AZURE_MODEL_VERSION,
            ModelKeyValues.USE_AZURE_OPENAI,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_azure_openai_embedding_save_and_unset_roundtrip(env_path: Path):
    """Verify: set-azure-openai-embedding --save writes keys; running again upserts; unset removes only those keys."""

    # save
    argv = build_args(
        "set-azure-openai-embedding",
        options={
            "--embedding-deployment-name": "fake-deployment",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)
    assert_env_contains(
        env_path,
        {
            EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME: "fake-deployment",
        },
    )

    assert_model_switched_to(
        env_path, EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING
    )

    # upsert without dupes
    argv = build_args(
        "set-azure-openai-embedding",
        options={
            "--embedding-deployment-name": "fake-deployment",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)
    assert_env_contains(
        env_path,
        {
            EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME: "fake-deployment",
        },
    )
    assert_no_dupes(
        env_path, [EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME]
    )

    # keep unrelated, unset only Azure keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-azure-openai-embedding", save_path=env_path)
    invoke_ok(app, argv)
    assert_env_lacks(
        env_path,
        [
            EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME,
            EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_ollama_model_save_and_unset(env_path: Path):
    """Verify: set-ollama --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-ollama",
        positionals=["ollama-model"],
        options={
            "--base-url": "https://fake-endpoint.example.com",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.LOCAL_MODEL_NAME: "ollama-model",
            ModelKeyValues.LOCAL_MODEL_BASE_URL: "https://fake-endpoint.example.com/",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.LOCAL_MODEL_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.LOCAL_MODEL_API_KEY: "ollama",
            ModelKeyValues.LOCAL_MODEL_NAME: "ollama-model",
            ModelKeyValues.LOCAL_MODEL_BASE_URL: "https://fake-endpoint.example.com/",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_LOCAL_MODEL)

    # keep unrelated, unset only local ollama keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-ollama", save_path=env_path)
    invoke_ok(app, argv)
    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.LOCAL_MODEL_NAME,
            ModelKeyValues.LOCAL_MODEL_BASE_URL,
            ModelKeyValues.LOCAL_MODEL_API_KEY,
            ModelKeyValues.USE_LOCAL_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_ollama_embedding_model_save_and_unset(env_path: Path):
    """Verify: set-ollama-embeddings --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-ollama-embeddings",
        positionals=["ollama-embedding-model"],
        options={
            "--base-url": "https://fake-endpoint.example.com",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: "ollama-embedding-model",
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: "https://fake-endpoint.example.com/",
        }
    )

    assert_deepeval_json_lacks([EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY])

    assert_env_contains(
        env_path,
        {
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY: "ollama",
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: "ollama-embedding-model",
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: "https://fake-endpoint.example.com/",
        },
    )

    assert_model_switched_to(env_path, EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)

    # keep unrelated, unset only local ollama embedding keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-ollama-embeddings", save_path=env_path)
    invoke_ok(app, argv)
    assert_env_lacks(
        env_path,
        [
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME,
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY,
            EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_local_model_save_and_unset(env_path: Path):
    """Verify: set-local-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-local-model",
        options={
            "--model-name": "local-model",
            "--base-url": "https://fake-endpoint.example.com",
            "--api-key": "local-key",
            "--format": "custom",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.LOCAL_MODEL_NAME: "local-model",
            ModelKeyValues.LOCAL_MODEL_BASE_URL: "https://fake-endpoint.example.com/",
            ModelKeyValues.LOCAL_MODEL_FORMAT: "custom",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.LOCAL_MODEL_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.LOCAL_MODEL_API_KEY: "local-key",
            ModelKeyValues.LOCAL_MODEL_NAME: "local-model",
            ModelKeyValues.LOCAL_MODEL_BASE_URL: "https://fake-endpoint.example.com/",
            ModelKeyValues.LOCAL_MODEL_FORMAT: "custom",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_LOCAL_MODEL)

    # keep unrelated, unset only local ollama keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-ollama", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.LOCAL_MODEL_NAME,
            ModelKeyValues.LOCAL_MODEL_BASE_URL,
            ModelKeyValues.LOCAL_MODEL_API_KEY,
            ModelKeyValues.USE_LOCAL_MODEL,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.LOCAL_MODEL_NAME,
            ModelKeyValues.LOCAL_MODEL_BASE_URL,
            ModelKeyValues.LOCAL_MODEL_API_KEY,
            ModelKeyValues.USE_LOCAL_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_local_model_does_not_write_none_api_key_when_missing(env_path: Path):
    argv = build_args(
        "set-local-model",
        options={
            "--model-name": "local-model",
            "--base-url": "http://localhost:8000",
            # no --api-key on purpose
            "--format": "json",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # JSON: format/model/base present; api key absent
    assert_deepeval_json_contains(
        {
            ModelKeyValues.LOCAL_MODEL_NAME: "local-model",
            ModelKeyValues.LOCAL_MODEL_BASE_URL: "http://localhost:8000/",
            ModelKeyValues.LOCAL_MODEL_FORMAT: "json",
        }
    )
    assert_deepeval_json_lacks([ModelKeyValues.LOCAL_MODEL_API_KEY])

    # dotenv: no API key line at all (not "None")
    assert_env_contains(
        env_path,
        {
            ModelKeyValues.LOCAL_MODEL_NAME: "local-model",
            ModelKeyValues.LOCAL_MODEL_BASE_URL: "http://localhost:8000/",
            ModelKeyValues.LOCAL_MODEL_FORMAT: "json",
        },
    )
    assert_env_lacks(
        env_path,
        [ModelKeyValues.LOCAL_MODEL_API_KEY],
    )


def test_grok_model_save_and_unset(env_path: Path):
    """Verify: set-grok-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-grok",
        options={
            "--model-name": "grok-model",
            "--api-key": "grok-key",
            "--temperature": 0.8,
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.GROK_MODEL_NAME: "grok-model",
            ModelKeyValues.TEMPERATURE: "0.8",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.GROK_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.GROK_API_KEY: "grok-key",
            ModelKeyValues.GROK_MODEL_NAME: "grok-model",
            ModelKeyValues.TEMPERATURE: "0.8",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_GROK_MODEL)

    # keep unrelated, unset only local ollama keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-grok", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.GROK_API_KEY,
            ModelKeyValues.GROK_MODEL_NAME,
            ModelKeyValues.TEMPERATURE,
            ModelKeyValues.USE_GROK_MODEL,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.GROK_API_KEY,
            ModelKeyValues.GROK_MODEL_NAME,
            ModelKeyValues.TEMPERATURE,
            ModelKeyValues.USE_GROK_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_moonshot_model_save_and_unset(env_path: Path):
    """Verify: set-moonshot-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-moonshot",
        options={
            "--model-name": "moonshot-model",
            "--api-key": "moonshot-key",
            "--temperature": 0.8,
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.MOONSHOT_MODEL_NAME: "moonshot-model",
            ModelKeyValues.TEMPERATURE: "0.8",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.MOONSHOT_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.MOONSHOT_API_KEY: "moonshot-key",
            ModelKeyValues.MOONSHOT_MODEL_NAME: "moonshot-model",
            ModelKeyValues.TEMPERATURE: "0.8",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_MOONSHOT_MODEL)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-moonshot", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.MOONSHOT_API_KEY,
            ModelKeyValues.MOONSHOT_MODEL_NAME,
            ModelKeyValues.TEMPERATURE,
            ModelKeyValues.USE_MOONSHOT_MODEL,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.MOONSHOT_API_KEY,
            ModelKeyValues.MOONSHOT_MODEL_NAME,
            ModelKeyValues.TEMPERATURE,
            ModelKeyValues.USE_MOONSHOT_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_deepseek_model_save_and_unset(env_path: Path):
    """Verify: set-deepseek-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-deepseek",
        options={
            "--model-name": "deepseek-model",
            "--api-key": "deepseek-key",
            "--temperature": 0.8,
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.DEEPSEEK_MODEL_NAME: "deepseek-model",
            ModelKeyValues.TEMPERATURE: "0.8",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.DEEPSEEK_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.DEEPSEEK_API_KEY: "deepseek-key",
            ModelKeyValues.DEEPSEEK_MODEL_NAME: "deepseek-model",
            ModelKeyValues.TEMPERATURE: "0.8",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_DEEPSEEK_MODEL)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-deepseek", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.DEEPSEEK_API_KEY,
            ModelKeyValues.DEEPSEEK_MODEL_NAME,
            ModelKeyValues.TEMPERATURE,
            ModelKeyValues.USE_DEEPSEEK_MODEL,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.DEEPSEEK_API_KEY,
            ModelKeyValues.DEEPSEEK_MODEL_NAME,
            ModelKeyValues.TEMPERATURE,
            ModelKeyValues.USE_DEEPSEEK_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_local_embeddings_save_and_unset(env_path: Path):
    """Verify: set-local-embeddings --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-local-embeddings",
        options={
            "--model-name": "local-embeddings-model",
            "--base-url": "https://fake-endpoint.example.com",
            "--api-key": "local-embeddings-key",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: "local-embeddings-model",
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: "https://fake-endpoint.example.com/",
        }
    )

    assert_deepeval_json_lacks([EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY])

    assert_env_contains(
        env_path,
        {
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY: "local-embeddings-key",
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: "local-embeddings-model",
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: "https://fake-endpoint.example.com/",
        },
    )

    assert_model_switched_to(env_path, EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-local-embeddings", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY,
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME,
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
            EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY,
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME,
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
            EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_local_embeddings_does_not_write_none_api_key_when_missing(
    env_path: Path,
):
    argv = build_args(
        "set-local-embeddings",
        options={
            "--model-name": "emb-model",
            "--base-url": "http://localhost:9000",
            # no --api-key
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    assert_deepeval_json_contains(
        {
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: "emb-model",
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: "http://localhost:9000/",
        }
    )
    assert_deepeval_json_lacks([EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY])

    assert_env_contains(
        env_path,
        {
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: "emb-model",
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: "http://localhost:9000/",
        },
    )
    assert_env_lacks(
        env_path,
        [EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY],
    )


def test_gemini_model_save_and_unset(env_path: Path):
    """Verify: set-gemini-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-gemini",
        options={
            "--model-name": "gemini-model",
            "--google-api-key": "google-gemini-api-key",
            "--project-id": "google-cloud-project-id",
            "--location": "google-cloud-location",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.GEMINI_MODEL_NAME: "gemini-model",
            ModelKeyValues.GOOGLE_CLOUD_PROJECT: "google-cloud-project-id",
            ModelKeyValues.GOOGLE_CLOUD_LOCATION: "google-cloud-location",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.GOOGLE_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.GOOGLE_API_KEY: "google-gemini-api-key",
            ModelKeyValues.GEMINI_MODEL_NAME: "gemini-model",
            ModelKeyValues.GOOGLE_CLOUD_PROJECT: "google-cloud-project-id",
            ModelKeyValues.GOOGLE_CLOUD_LOCATION: "google-cloud-location",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_GEMINI_MODEL)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-gemini", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.GOOGLE_API_KEY,
            ModelKeyValues.GEMINI_MODEL_NAME,
            ModelKeyValues.USE_GEMINI_MODEL,
            ModelKeyValues.GOOGLE_CLOUD_PROJECT,
            ModelKeyValues.GOOGLE_CLOUD_LOCATION,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.GOOGLE_API_KEY,
            ModelKeyValues.GEMINI_MODEL_NAME,
            ModelKeyValues.USE_GEMINI_MODEL,
            ModelKeyValues.GOOGLE_CLOUD_PROJECT,
            ModelKeyValues.GOOGLE_CLOUD_LOCATION,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_gemini_model_save_without_model_set_unset(env_path: Path):
    """Verify: set-gemini-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-gemini",
        options={
            "--google-api-key": "google-gemini-api-key",
            "--project-id": "google-cloud-project-id",
            "--location": "google-cloud-location",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.GOOGLE_CLOUD_PROJECT: "google-cloud-project-id",
            ModelKeyValues.GOOGLE_CLOUD_LOCATION: "google-cloud-location",
        }
    )

    assert_deepeval_json_lacks(
        [ModelKeyValues.GEMINI_MODEL_NAME, ModelKeyValues.GOOGLE_API_KEY]
    )

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.GOOGLE_API_KEY: "google-gemini-api-key",
            ModelKeyValues.GOOGLE_CLOUD_PROJECT: "google-cloud-project-id",
            ModelKeyValues.GOOGLE_CLOUD_LOCATION: "google-cloud-location",
        },
    )

    assert_env_lacks(env_path, [ModelKeyValues.GEMINI_MODEL_NAME])

    assert_model_switched_to(env_path, ModelKeyValues.USE_GEMINI_MODEL)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-gemini", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.GOOGLE_API_KEY,
            ModelKeyValues.GEMINI_MODEL_NAME,
            ModelKeyValues.GOOGLE_CLOUD_PROJECT,
            ModelKeyValues.GOOGLE_CLOUD_LOCATION,
            ModelKeyValues.USE_GEMINI_MODEL,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.GOOGLE_API_KEY,
            ModelKeyValues.GEMINI_MODEL_NAME,
            ModelKeyValues.GOOGLE_CLOUD_PROJECT,
            ModelKeyValues.GOOGLE_CLOUD_LOCATION,
            ModelKeyValues.USE_GEMINI_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_gemini_model_save_without_api_key_set_unset(env_path: Path):
    """Verify: set-gemini-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-gemini",
        options={
            "--project-id": "google-cloud-project-id",
            "--location": "google-cloud-location",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.GOOGLE_CLOUD_PROJECT: "google-cloud-project-id",
            ModelKeyValues.GOOGLE_CLOUD_LOCATION: "google-cloud-location",
        }
    )

    assert_deepeval_json_lacks(
        [ModelKeyValues.GEMINI_MODEL_NAME, ModelKeyValues.GOOGLE_API_KEY]
    )

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.GOOGLE_CLOUD_PROJECT: "google-cloud-project-id",
            ModelKeyValues.GOOGLE_CLOUD_LOCATION: "google-cloud-location",
        },
    )

    assert_env_lacks(
        env_path,
        [ModelKeyValues.GEMINI_MODEL_NAME, ModelKeyValues.GOOGLE_API_KEY],
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_GEMINI_MODEL)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-gemini", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.GOOGLE_API_KEY,
            ModelKeyValues.GEMINI_MODEL_NAME,
            ModelKeyValues.GOOGLE_CLOUD_PROJECT,
            ModelKeyValues.GOOGLE_CLOUD_LOCATION,
            ModelKeyValues.USE_GEMINI_MODEL,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.GOOGLE_API_KEY,
            ModelKeyValues.GEMINI_MODEL_NAME,
            ModelKeyValues.GOOGLE_CLOUD_PROJECT,
            ModelKeyValues.GOOGLE_CLOUD_LOCATION,
            ModelKeyValues.USE_GEMINI_MODEL,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_set_gemini_enforces_api_key_or_project_location(env_path: Path):
    """Verify: set-gemini-model without the api key and either project or location raises error."""

    invalid_option_combinations = [
        {"--location": "google-cloud-location"},
        {
            "--project-id": "google-cloud-project-id",
        },
        {},
    ]

    for options in invalid_option_combinations:
        argv = build_args(
            "set-gemini",
            options=options,
            save_path=env_path,
        )
        invoke_with_error(
            app,
            argv,
            expected_error="You must provide either --google-api-key or both --project-id and --location.",
        )


def test_litellm_model_save_and_unset(env_path: Path):
    """Verify: set-litellm-model --save writes expected keys; unset removes only those."""
    # Setup
    argv = build_args(
        "set-litellm",
        positionals=["litellm-model"],
        options={
            "--api-key": "litellm-api-key",
            "--api-base": "https://fake-endpoint.example.com",
        },
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Assert that variables persisted as expected
    assert_deepeval_json_contains(
        {
            ModelKeyValues.LITELLM_MODEL_NAME: "litellm-model",
            ModelKeyValues.LITELLM_API_BASE: "https://fake-endpoint.example.com/",
        }
    )

    assert_deepeval_json_lacks([ModelKeyValues.LITELLM_API_KEY])

    assert_env_contains(
        env_path,
        {
            ModelKeyValues.LITELLM_API_KEY: "litellm-api-key",
            ModelKeyValues.LITELLM_MODEL_NAME: "litellm-model",
            ModelKeyValues.LITELLM_API_BASE: "https://fake-endpoint.example.com/",
        },
    )

    assert_model_switched_to(env_path, ModelKeyValues.USE_LITELLM)

    # keep unrelated, unset only moonshot keys
    with env_path.open("a") as f:
        f.write("\nUNRELATED_KEY=keepme\n")

    argv = build_args("unset-litellm", save_path=env_path)
    invoke_ok(app, argv)

    assert_deepeval_json_lacks(
        [
            ModelKeyValues.LITELLM_API_KEY,
            ModelKeyValues.LITELLM_MODEL_NAME,
            ModelKeyValues.LITELLM_API_BASE,
        ]
    )

    assert_env_lacks(
        env_path,
        [
            ModelKeyValues.LITELLM_API_KEY,
            ModelKeyValues.LITELLM_MODEL_NAME,
            ModelKeyValues.LITELLM_API_BASE,
        ],
    )
    env = read_dotenv_as_dict(env_path)
    assert env.get("UNRELATED_KEY") == "keepme"


def test_login_with_confident_api_key_implies_dotenv_save(env_path: Path):
    """--confident-api-key should persist to dotenv by default; should not hit JSON secrets."""
    argv = build_args(
        "login",
        options={"--confident-api-key": "ck-test-123"},  # no --save provided
        save_path=env_path,  # build_args wires default save target; we still want to verify content goes here
    )
    invoke_ok(app, argv)

    # dotenv: key present
    assert_env_contains(
        env_path,
        {
            KeyValues.API_KEY.value.upper(): "ck-test-123",
        },
    )

    # JSON: key must not be written (secrets never go to JSON)
    assert_deepeval_json_lacks([KeyValues.API_KEY])


def test_login_with_confident_api_key_honors_custom_save_path(tmp_path: Path):
    custom_env = tmp_path / ".myconf.env"
    argv = build_args(
        "login",
        options={"--confident-api-key": "ck-custom"},
        save_path=custom_env,
    )
    invoke_ok(app, argv)

    assert_env_contains(
        custom_env,
        {KeyValues.API_KEY.value.upper(): "ck-custom"},
    )


def test_logout_removes_dotenv_and_json_by_default(env_path: Path):
    """logout should remove from dotenv (default path) and JSON store without requiring --save."""
    # Login
    argv = build_args(
        "login",
        options={"--confident-api-key": "ck-test-xyz"},
        save_path=env_path,
    )
    invoke_ok(app, argv)

    # Sanity: login wrote api_key into dotenv; nothing in JSON
    assert_env_contains(
        env_path, {KeyValues.API_KEY.value.upper(): "ck-test-xyz"}
    )
    assert_deepeval_json_lacks([KeyValues.API_KEY])

    # Logout (no --save flag; should still remove from dotenv + JSON)
    argv = build_args("logout", save_path=env_path)
    invoke_ok(app, argv)

    # Dotenv no longer contains either alias
    env_after = read_dotenv_as_dict(env_path)
    assert "api_key" not in env_after
    assert "CONFIDENT_API_KEY" not in env_after

    # JSON cleared
    assert_deepeval_json_lacks([KeyValues.API_KEY])


def test_logout_honors_custom_save_path(tmp_path: Path):
    """logout should remove keys from a custom dotenv path when provided."""
    custom_env = tmp_path / ".myconf.env"

    # Login to custom path
    argv = build_args(
        "login",
        options={"--confident-api-key": "ck-custom"},
        save_path=custom_env,
    )
    invoke_ok(app, argv)
    assert_env_contains(
        custom_env, {KeyValues.API_KEY.value.upper(): "ck-custom"}
    )
    assert_deepeval_json_lacks([KeyValues.API_KEY])

    # Logout using the same custom path
    argv = build_args("logout", save_path=custom_env)
    invoke_ok(app, argv)

    env_after = read_dotenv_as_dict(custom_env)
    assert "api_key" not in env_after
    assert "CONFIDENT_API_KEY" not in env_after
    assert_deepeval_json_lacks([KeyValues.API_KEY])


def test_login_then_logout_roundtrip_default_path(env_path: Path):
    """Roundtrip: login writes to dotenv (default), logout removes from dotenv and JSON."""

    # Login
    argv = build_args(
        "login",
        options={"--confident-api-key": "ck-roundtrip"},
        save_path=env_path,
    )
    invoke_ok(app, argv)
    assert_env_contains(
        env_path, {KeyValues.API_KEY.value.upper(): "ck-roundtrip"}
    )
    assert_deepeval_json_lacks([KeyValues.API_KEY])

    # Logout
    argv = build_args("logout", save_path=env_path)
    invoke_ok(app, argv)
    env_after = read_dotenv_as_dict(env_path)
    assert "api_key" not in env_after
    assert "CONFIDENT_API_KEY" not in env_after
    assert_deepeval_json_lacks([KeyValues.API_KEY])


def test_logout_is_idempotent(env_path: Path):
    """Calling logout when nothing is set should succeed and not create files."""

    argv = build_args("logout", save_path=env_path)
    invoke_ok(app, argv)

    # No dotenv created and JSON lacks the key
    assert not env_path.exists()
    assert_deepeval_json_lacks([KeyValues.API_KEY])
