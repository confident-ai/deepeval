import os
from pathlib import Path
import pytest

from deepeval.config.utils import parse_bool
from deepeval.config.settings import (
    autoload_dotenv,
    get_settings,
    reset_settings,
)


@pytest.mark.enable_dotenv
def test_autoload_dotenv_precedence(monkeypatch, env_dir: Path):
    # .env sets base, .env.dev overrides, .env.local highest
    (env_dir / ".env").write_text("APP_ENV=dev\nFOO=base\n")
    (env_dir / ".env.dev").write_text("FOO=env\n")
    (env_dir / ".env.local").write_text("FOO=local\n")

    autoload_dotenv()
    assert os.environ["APP_ENV"] == "dev"
    assert os.environ["FOO"] == "local"  # local wins
    monkeypatch.delenv("FOO", raising=False)


@pytest.mark.enable_dotenv
def test_autoload_respects_disable_flag(monkeypatch, env_dir: Path):
    (env_dir / ".env").write_text("FOO=base\n")
    monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "1")
    autoload_dotenv()
    assert "FOO" not in os.environ  # skipped


@pytest.mark.enable_dotenv
def test_autoload_does_not_override_process_env(monkeypatch, env_dir: Path):
    (env_dir / ".env").write_text("FOO=base\n")
    monkeypatch.setenv("FOO", "proc")  # process env wins
    autoload_dotenv()
    assert os.environ["FOO"] == "proc"


@pytest.mark.enable_dotenv
def test_autoload_respects_env_dir_path(monkeypatch, tmp_path: Path):
    env_dir = tmp_path / "custom"
    env_dir.mkdir()
    (env_dir / ".env.local").write_text("FROM_CUSTOM_DIR=1\n")
    monkeypatch.setenv("ENV_DIR_PATH", str(env_dir))
    autoload_dotenv()
    assert os.environ.get("FROM_CUSTOM_DIR") == "1"


def test_defaults():
    # env is cleared by conftest.
    # should see model defaults
    s = get_settings()
    assert s.CONFIDENT_TRACE_VERBOSE is True
    assert s.CONFIDENT_TRACE_SAMPLE_RATE == 1.0
    assert s.CONFIDENT_METRIC_LOGGING_VERBOSE is True
    assert s.CONFIDENT_METRIC_LOGGING_SAMPLE_RATE == 1.0
    assert s.CONFIDENT_METRIC_LOGGING_ENABLED is True


def test_env_mutation_after_init_triggers_auto_refresh(monkeypatch):
    from deepeval.config.settings import get_settings

    s1 = get_settings()
    assert s1.USE_OPENAI_MODEL in (None, False, True)

    monkeypatch.setenv("USE_OPENAI_MODEL", "YES")
    s2 = get_settings()
    assert s2 is not s1  # should auto refresh when env updates
    assert s2.USE_OPENAI_MODEL is True


def test_invalid_trace_sample_rate_raises(monkeypatch):
    # set env before first construction to trigger the validator
    monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", "1.2")
    with pytest.raises(ValueError):
        get_settings()


def test_invalid_metric_sample_rate_raises(monkeypatch):
    # set env before first construction to trigger the validator
    monkeypatch.setenv("CONFIDENT_METRIC_LOGGING_SAMPLE_RATE", "1.2")
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


@pytest.mark.enable_dotenv
def test_edit_respects_default_save_writes_dotenv(monkeypatch, env_dir: Path):
    # configure default save to a specific file
    dotenv_path = env_dir / ".env"
    monkeypatch.setenv("DEEPEVAL_DEFAULT_SAVE", f"dotenv:{dotenv_path}")

    s = get_settings()
    with s.edit():  # uses DEEPEVAL_DEFAULT_SAVE
        s.GRPC_VERBOSITY = "ERROR"

    assert dotenv_path.exists()
    content = dotenv_path.read_text()
    assert "GRPC_VERBOSITY=ERROR" in content


@pytest.mark.enable_dotenv
def test_edit_explicit_save_overrides_default(monkeypatch, env_dir: Path):
    monkeypatch.setenv(
        "DEEPEVAL_DEFAULT_SAVE", f"dotenv:{env_dir / 'ignored.env'}"
    )
    explicit = env_dir / "chosen.env"

    s = get_settings()
    with s.edit(save=f"dotenv:{explicit}"):
        s.TOKENIZERS_PARALLELISM = True

    assert explicit.exists()
    assert "TOKENIZERS_PARALLELISM=1" in explicit.read_text()
    # and the default file was not created
    assert not (env_dir / "ignored.env").exists()


def test_switch_model_provider_flips_only_target():
    s = get_settings()
    with s.edit(persist=False) as ctx:
        # seed a couple of toggles
        s.USE_OPENAI_MODEL = False
        s.USE_LOCAL_MODEL = True
        ctx.switch_model_provider("USE_OPENAI_MODEL")

    assert s.USE_OPENAI_MODEL is True
    assert s.USE_LOCAL_MODEL is False


@pytest.mark.enable_dotenv
def test_edit_unset_removes_from_env_and_dotenv(monkeypatch, env_dir: Path):
    dotenv_path = env_dir / ".env"
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


def test_env_dir_path_expanduser(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENV_DIR_PATH", "~/envdir")
    s = get_settings()
    assert s.ENV_DIR_PATH == tmp_path / "envdir"


def test_results_folder_expandvars(monkeypatch, tmp_path: Path):
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
    "opt_out,expected",
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
@pytest.mark.enable_dotenv
def test_boolean_coercion_opt_in_with_autoload_dotenv(
    monkeypatch, env_path: Path, opt_out, expected
):
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    env_path.write_text(f"DEEPEVAL_TELEMETRY_OPT_OUT={opt_out}\n")
    autoload_dotenv()
    settings = get_settings()
    assert parse_bool(os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"]) is expected
    assert settings.DEEPEVAL_TELEMETRY_OPT_OUT is expected


@pytest.mark.parametrize(
    "opt_out,expected",
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
def test_boolean_coercion_opt_out_with_dotenv(monkeypatch, opt_out, expected):
    s = get_settings()
    with s.edit(persist=False):
        s.DEEPEVAL_TELEMETRY_OPT_OUT = opt_out
    assert s.DEEPEVAL_TELEMETRY_OPT_OUT is expected


def test_boolean_reset_settings_after_environ_update(monkeypatch):
    monkeypatch.setenv("USE_OPENAI_MODEL", "YES")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")

    settings = get_settings()
    assert settings.USE_OPENAI_MODEL is True
    assert settings.CUDA_LAUNCH_BLOCKING is False


def test_sample_rate_empty_string_is_none(monkeypatch):
    monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", "")
    s = get_settings()
    assert s.CONFIDENT_TRACE_SAMPLE_RATE is None
    monkeypatch.setenv("CONFIDENT_METRIC_LOGGING_SAMPLE_RATE", "")
    s = get_settings()
    assert s.CONFIDENT_METRIC_LOGGING_SAMPLE_RATE is None


@pytest.mark.parametrize("val", ["0", "1", "0.25"])
def test_sample_rate_valid_boundaries(monkeypatch, val):
    monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", val)
    s = get_settings()
    assert s.CONFIDENT_TRACE_SAMPLE_RATE == float(val)
    monkeypatch.setenv("CONFIDENT_METRIC_LOGGING_SAMPLE_RATE", val)
    s = get_settings()
    assert s.CONFIDENT_METRIC_LOGGING_SAMPLE_RATE == float(val)


@pytest.mark.parametrize("val", ["1.5", "-0.1"])
def test_sample_rate_invalid_raises(monkeypatch, val):
    monkeypatch.setenv("CONFIDENT_TRACE_SAMPLE_RATE", val)
    with pytest.raises(ValueError):
        get_settings()
    monkeypatch.setenv("CONFIDENT_METRIC_LOGGING_SAMPLE_RATE", val)
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


############################################################
# DEEPEVAL_TELEMETRY_ENABLED -> alias for *_OPT_OUT (secure)
############################################################


def _clear_telemetry_env(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_ENABLED", raising=False)


def test_alias_only_enabled_yes_sets_opt_out_false(monkeypatch):
    _clear_telemetry_env(monkeypatch)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_ENABLED", "YES")
    reset_settings(reload_dotenv=False)

    s = get_settings()
    assert s.DEEPEVAL_TELEMETRY_OPT_OUT is False  # ON


def test_alias_only_enabled_no_sets_opt_out_true(monkeypatch):
    _clear_telemetry_env(monkeypatch)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_ENABLED", "no")
    reset_settings(reload_dotenv=False)

    settings = get_settings()
    assert settings.DEEPEVAL_TELEMETRY_OPT_OUT is True


def test_alias_both_present_opt_out_wins(monkeypatch):
    # Conflict: OPT_OUT says OFF, legacy says ON means OFF wins
    _clear_telemetry_env(monkeypatch)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")  # OFF
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_ENABLED", "YES")  # ON
    reset_settings(reload_dotenv=False)

    settings = get_settings()
    assert settings.DEEPEVAL_TELEMETRY_OPT_OUT is True


def test_alias_both_present_enabled_false_forces_opt_out(monkeypatch):
    # Conflict: OPT_OUT says ON, legacy says OFF means OFF wins
    _clear_telemetry_env(monkeypatch)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "no")  # ON
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_ENABLED", "0")  # OFF
    reset_settings(reload_dotenv=False)

    settings = get_settings()
    assert settings.DEEPEVAL_TELEMETRY_OPT_OUT is True


def test_neither_set_defaults_on(monkeypatch):
    # neither var present means default OFF (for security)
    _clear_telemetry_env(monkeypatch)
    reset_settings(reload_dotenv=False)

    settings = get_settings()
    assert settings.DEEPEVAL_TELEMETRY_OPT_OUT is False  # ON by default


##################################################
# Do not persist DEEPEVAL_TELEMETRY_ENABLED
##################################################


@pytest.mark.enable_dotenv
def test_legacy_enabled_alias_not_persisted_to_dotenv(
    monkeypatch, env_dir: Path
):
    """
    We persist DEEPEVAL_TELEMETRY_OPT_OUT, but never the legacy DEEPEVAL_TELEMETRY_ENABLED.
    """
    dotenv_path = env_dir / ".env"
    monkeypatch.setenv("DEEPEVAL_DEFAULT_SAVE", f"dotenv:{dotenv_path}")

    _clear_telemetry_env(monkeypatch)
    # Seed legacy alias to YES so OPT_OUT starts as False.
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_ENABLED", "YES")
    reset_settings(reload_dotenv=False)

    settings = get_settings()
    # _ENABLED is ON -> OPT_OUT False
    assert settings.DEEPEVAL_TELEMETRY_OPT_OUT is False

    with settings.edit() as ctx:
        # Now flip it so a diff is recorded and persisted
        settings.DEEPEVAL_TELEMETRY_OPT_OUT = True  # OFF
        settings.DEEPEVAL_VERBOSE_MODE = True

    assert ctx.result is not None
    updated = ctx.result.updated

    # Legacy DEEPEVAL_TELEMETRY_ENABLED must not appear in the persisted updates
    assert "DEEPEVAL_TELEMETRY_ENABLED" not in updated
    # but other fields should
    assert "DEEPEVAL_TELEMETRY_OPT_OUT" in updated
    assert "DEEPEVAL_VERBOSE_MODE" in updated

    # Dotenv should not contain DEEPEVAL_TELEMETRY_ENABLED
    content = dotenv_path.read_text()
    assert "DEEPEVAL_TELEMETRY_ENABLED" not in content
    # Booleans are persisted as 1 or 0
    assert "DEEPEVAL_TELEMETRY_OPT_OUT=1" in content
    assert "DEEPEVAL_VERBOSE_MODE=1" in content
