import os
from pathlib import Path
import pytest

from deepeval.config.settings import autoload_dotenv, get_settings

pytestmark = pytest.mark.skip(
    reason="Temporarily disabled while refactoring settings persistence"
)


def test_autoload_dotenv_precedence(tmp_path: Path, monkeypatch):
    # .env sets base, .env.dev overrides, .env.local highest
    (tmp_path / ".env").write_text("APP_ENV=dev\nFOO=base\n")
    (tmp_path / ".env.dev").write_text("FOO=env\n")
    (tmp_path / ".env.local").write_text("FOO=local\n")

    autoload_dotenv()
    assert os.environ["APP_ENV"] == "dev"
    assert os.environ["FOO"] == "local"  # local wins
    monkeypatch.delenv("FOO", raising=False)


def test_autoload_respects_disable_flag(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("FOO=base\n")
    monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "1")
    monkeypatch.delenv(
        "FOO", raising=False
    )  # cleanup from past tests. will find a better way
    autoload_dotenv()
    assert "FOO" not in os.environ  # skipped


def test_autoload_does_not_override_process_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FOO=base\n")
    monkeypatch.setenv("FOO", "proc")  # process env wins
    autoload_dotenv()
    assert os.environ["FOO"] == "proc"


def test_autoload_respects_env_dir_path(tmp_path, monkeypatch):
    env_dir = tmp_path / "custom"
    env_dir.mkdir()
    (env_dir / ".env.local").write_text("FROM_CUSTOM_DIR=1\n")
    monkeypatch.setenv("ENV_DIR_PATH", str(env_dir))
    autoload_dotenv()
    assert os.environ.get("FROM_CUSTOM_DIR") == "1"


def test_boolean_coercion_yes_no_and_10(monkeypatch):
    monkeypatch.setenv("USE_OPENAI_MODEL", "YES")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")

    s = get_settings()
    assert s.USE_OPENAI_MODEL is True
    assert s.CUDA_LAUNCH_BLOCKING is False


def test_defaults():
    # env is cleared by conftest.
    # should see model defaults
    s = get_settings()
    assert s.CONFIDENT_TRACE_VERBOSE is True
    assert s.CONFIDENT_SAMPLE_RATE == 1.0


def test_invalid_sample_rate_raises(monkeypatch):
    # set env before first construction to trigger the validator
    monkeypatch.setenv("CONFIDENT_SAMPLE_RATE", "1.2")
    with pytest.raises(ValueError):
        get_settings()


def test_edit_runtime_only_persist_false_updates_env_not_files(
    tmp_path: Path, monkeypatch
):
    # spy on legacy JSON to ensure no writes when persist=False
    import deepeval.key_handler as key_handler_mod

    writes = []
    monkeypatch.setattr(
        key_handler_mod.KEY_FILE_HANDLER,
        "write_key",
        lambda k, v: writes.append((k, v)),
    )
    monkeypatch.setattr(
        key_handler_mod.KEY_FILE_HANDLER,
        "remove_key",
        lambda k: writes.append((k, None)),
    )

    s = get_settings()
    with s.edit(persist=False):
        s.DEEPEVAL_IDENTIFIER = "abc123"
        s.USE_OPENAI_MODEL = True

    # runtime env reflects changes
    assert os.environ.get("DEEPEVAL_IDENTIFIER") == "abc123"
    assert os.environ.get("USE_OPENAI_MODEL") == "1"

    # no files created and no JSON writes
    assert not any(p.name.startswith(".env") for p in tmp_path.iterdir())
    assert writes == []


def test_edit_respects_default_save_writes_dotenv(tmp_path: Path, monkeypatch):
    # configure default save to a specific file
    dotenv_path = tmp_path / ".env"
    monkeypatch.setenv("DEEPEVAL_DEFAULT_SAVE", f"dotenv:{dotenv_path}")

    s = get_settings()
    with s.edit():  # uses DEEPEVAL_DEFAULT_SAVE
        s.GRPC_VERBOSITY = "ERROR"

    assert dotenv_path.exists()
    content = dotenv_path.read_text()
    assert "GRPC_VERBOSITY=ERROR" in content


def test_edit_explicit_save_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DEEPEVAL_DEFAULT_SAVE", f"dotenv:{tmp_path / 'ignored.env'}"
    )
    explicit = tmp_path / "chosen.env"

    s = get_settings()
    with s.edit(save=f"dotenv:{explicit}"):
        s.TOKENIZERS_PARALLELISM = True

    assert explicit.exists()
    assert "TOKENIZERS_PARALLELISM=1" in explicit.read_text()
    # and the default file was not created
    assert not (tmp_path / "ignored.env").exists()


def test_switch_model_provider_flips_only_target():
    s = get_settings()
    with s.edit(persist=False) as ctx:
        # seed a couple of toggles
        s.USE_OPENAI_MODEL = False
        s.USE_LOCAL_MODEL = True
        ctx.switch_model_provider("USE_OPENAI_MODEL")

    assert s.USE_OPENAI_MODEL is True
    assert s.USE_LOCAL_MODEL is False


def test_edit_unset_removes_from_env_and_dotenv(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    monkeypatch.setenv("DEEPEVAL_DEFAULT_SAVE", f"dotenv:{dotenv_path}")

    # seed a value via settings so it ends up in dotenv
    s = get_settings()
    with s.edit():  # default save should persist
        s.GRPC_VERBOSITY = "ERROR"
    assert "GRPC_VERBOSITY=ERROR" in dotenv_path.read_text()

    # now unset it and ensure it’s removed everywhere
    with s.edit():
        s.GRPC_VERBOSITY = None
    assert "GRPC_VERBOSITY" not in os.environ
    assert "GRPC_VERBOSITY" not in dotenv_path.read_text()


def test_secret_not_persisted_to_json(monkeypatch):
    # spy on legacy JSON methods
    import deepeval.key_handler as key_handler_mod

    calls = []
    monkeypatch.setattr(
        key_handler_mod.KEY_FILE_HANDLER,
        "write_key",
        lambda k, v: calls.append(("write", k, v)),
    )
    monkeypatch.setattr(
        key_handler_mod.KEY_FILE_HANDLER,
        "remove_key",
        lambda k: calls.append(("remove", k)),
    )

    s = get_settings()
    from pydantic import SecretStr

    with s.edit(
        persist=True
    ):  # allow JSON/dotenv, but secrets should be skipped for JSON
        s.OPENAI_API_KEY = SecretStr("sk-abc123")

    # no JSON writes for a SecretStr field
    assert not calls


def test_env_dir_path_expanduser(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENV_DIR_PATH", "~/envdir")
    s = get_settings()
    assert s.ENV_DIR_PATH == tmp_path / "envdir"


def test_results_folder_expandvars(tmp_path, monkeypatch):
    outdir = tmp_path / "outdir"
    monkeypatch.setenv("MYDIR", str(outdir))
    monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", "$MYDIR")
    s = get_settings()
    assert s.DEEPEVAL_RESULTS_FOLDER == outdir


def test_env_dir_path_empty_string_is_none(monkeypatch):
    monkeypatch.setenv("ENV_DIR_PATH", "")
    s = get_settings()
    assert s.ENV_DIR_PATH is None


@pytest.mark.parametrize("val", ["readonly", "Read-Only", "READONLY", "RO"])
def test_filesystem_aliases_normalized(monkeypatch, val):
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", val)
    s = get_settings()
    assert s.DEEPEVAL_FILE_SYSTEM == "READ_ONLY"


def test_filesystem_invalid_raises(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "WRITABLE")
    with pytest.raises(ValueError):
        get_settings()


@pytest.mark.parametrize(
    "tok,expected",
    [
        ("YES", True),
        ("No", False),
        ("1", True),
        ("0", False),
        ("on", True),
        ("off", False),
        ("enable", True),
        ("disabled", False),
    ],
)
def test_boolean_coercion_tokens(monkeypatch, tok, expected):
    # Use a representative boolean field
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", tok)
    s = get_settings()
    assert s.TOKENIZERS_PARALLELISM is expected


def test_sample_rate_empty_string_is_none(monkeypatch):
    monkeypatch.setenv("CONFIDENT_SAMPLE_RATE", "")
    s = get_settings()
    assert s.CONFIDENT_SAMPLE_RATE is None


@pytest.mark.parametrize("val", ["0", "1", "0.25"])
def test_sample_rate_valid_boundaries(monkeypatch, val):
    monkeypatch.setenv("CONFIDENT_SAMPLE_RATE", val)
    s = get_settings()
    assert s.CONFIDENT_SAMPLE_RATE == float(val)


@pytest.mark.parametrize("val", ["1.5", "-0.1"])
def test_sample_rate_invalid_raises(monkeypatch, val):
    monkeypatch.setenv("CONFIDENT_SAMPLE_RATE", val)
    with pytest.raises(ValueError):
        get_settings()


def test_switch_model_provider_flips_all_use_flags(monkeypatch):
    s = get_settings()
    with s.edit(persist=False) as ctx:
        # Seed a mix of USE_* flags across models/embeddings
        for field in type(s).model_fields:
            if field.startswith("USE_"):
                setattr(s, field, True)
        ctx.switch_model_provider("USE_GEMINI_MODEL")

    for field in type(s).model_fields:
        if field.startswith("USE_"):
            if field == "USE_GEMINI_MODEL":
                assert getattr(s, field) is True
            else:
                assert getattr(s, field) is False
