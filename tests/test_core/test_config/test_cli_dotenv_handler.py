import json
import os
import stat
from pathlib import Path

import pytest
from dotenv import dotenv_values
from typer import BadParameter

from deepeval.cli.dotenv_handler import DotenvHandler, DotenvTargetError


@pytest.fixture
def reset_settings_after_test():
    yield
    from deepeval.config.settings import reset_settings

    reset_settings(reload_dotenv=False)


def _count_key_occurrences(content: str, key: str) -> int:
    return sum(1 for line in content.splitlines() if line.startswith(f"{key}="))


def test_upsert_round_trips_values_that_need_dotenv_escaping(tmp_path):
    dotenv_path = tmp_path / "nested" / ".env.local"
    values = {
        "SIMPLE": "ERROR",
        "SPACE_HASH": "hello # world",
        "DOUBLE_QUOTE": 'token with " quote',
        "SINGLE_QUOTE": "token with ' quote",
        "NEWLINE": "line1\nline2",
        "BACKSLASH": r"C:\tmp\value with space",
        "TRAILING_BACKSLASH": "C:\\models\\",
        "TRAILING_BACKSLASH_WITH_SPACE": "C:\\model dir\\",
        "UNC_TRAILING_BACKSLASH": "\\\\server\\share\\",
    }

    DotenvHandler(dotenv_path).upsert(values)

    assert dict(dotenv_values(dotenv_path)) == values
    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600


def test_upsert_keeps_shell_safe_values_bare_for_compatibility(tmp_path):
    dotenv_path = tmp_path / ".env.local"
    values = {
        "OPENAI_API_KEY": "sk-proj_abc-123",
        "SERVICE_ACCOUNT_PATH": "/tmp/key-file.json",
        "EMPTY": "",
    }

    DotenvHandler(dotenv_path).upsert(values)

    content = dotenv_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-proj_abc-123\n" in content
    assert "SERVICE_ACCOUNT_PATH=/tmp/key-file.json\n" in content
    assert "EMPTY=\n" in content
    assert dict(dotenv_values(dotenv_path)) == values


def test_upsert_round_trips_literal_interpolation_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/expanded-home")
    monkeypatch.setenv("TOKEN", "expanded-token")
    dotenv_path = tmp_path / ".env.local"
    values = {
        "HOME_TEMPLATE": "${HOME}",
        "PREFIXED_TEMPLATE": "prefix-${TOKEN}-suffix",
        "DEFAULT_TEMPLATE": "${MISSING:-fallback}",
        "SPACED_TEMPLATE": "hello ${HOME}",
    }

    DotenvHandler(dotenv_path).upsert(values)

    content = dotenv_path.read_text(encoding="utf-8")
    assert "${:-$}{HOME}" in content
    assert "${:-$}{TOKEN}" in content
    assert dict(dotenv_values(dotenv_path)) == values


def test_upsert_preserves_comments_and_updates_without_duplicates(tmp_path):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "# model settings\n" "EXISTING=old\n" "KEEP=unchanged\n",
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).upsert(
        {
            "EXISTING": "updated # with comment marker",
            "ADDED": r"C:\tmp\added value",
        }
    )

    content = dotenv_path.read_text(encoding="utf-8")
    parsed = dotenv_values(dotenv_path)

    assert content.startswith("# model settings\n")
    assert "KEEP=unchanged\n" in content
    assert _count_key_occurrences(content, "EXISTING") == 1
    assert parsed["EXISTING"] == "updated # with comment marker"
    assert parsed["ADDED"] == r"C:\tmp\added value"
    assert parsed["KEEP"] == "unchanged"


def test_unset_removes_keys_while_leaving_other_lines_intact(tmp_path):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "# keep this comment\n" "REMOVE=secret\n" "KEEP=value\n",
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).unset(["REMOVE", "MISSING"])

    content = dotenv_path.read_text(encoding="utf-8")
    assert content.startswith("# keep this comment\n")
    assert "REMOVE=" not in content
    assert "KEEP=value\n" in content
    assert dotenv_values(dotenv_path)["KEEP"] == "value"


def test_unset_only_missing_regular_dotenv_stays_silent(tmp_path):
    dotenv_path = tmp_path / ".env.local"

    DotenvHandler(dotenv_path).unset(["MISSING"])

    assert not dotenv_path.exists()


def test_upsert_replaces_legacy_malformed_assignment_without_duplicate(
    tmp_path,
):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        'BROKEN="token with " quote"\n' "KEEP=unchanged\n",
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).upsert({"BROKEN": 'fixed " value'})

    content = dotenv_path.read_text(encoding="utf-8")
    parsed = dotenv_values(dotenv_path)

    assert _count_key_occurrences(content, "BROKEN") == 1
    assert parsed["BROKEN"] == 'fixed " value'
    assert parsed["KEEP"] == "unchanged"


@pytest.mark.parametrize("key", ["FOO.BAR", "A-B", "1A"])
def test_upsert_replaces_malformed_assignments_for_dotenv_key_grammar(
    tmp_path,
    key,
):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        f'{key}="token with " quote"\nKEEP=unchanged\n',
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).upsert({key: "fixed"})

    content = dotenv_path.read_text(encoding="utf-8")
    parsed = dotenv_values(dotenv_path)

    assert _count_key_occurrences(content, key) == 1
    assert parsed[key] == "fixed"
    assert parsed["KEEP"] == "unchanged"


def test_unset_removes_legacy_malformed_assignment_silently(tmp_path, caplog):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        'BROKEN="token with " quote"\n' "KEEP=unchanged\n",
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).unset(["BROKEN", "MISSING"])

    content = dotenv_path.read_text(encoding="utf-8")
    assert "BROKEN=" not in content
    assert "KEEP=unchanged\n" in content
    assert not [
        record for record in caplog.records if record.name.startswith("dotenv")
    ]


def test_unset_removes_malformed_assignment_for_dotenv_key_grammar(tmp_path):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        'FOO.BAR="token with " quote"\nKEEP=unchanged\n',
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).unset(["FOO.BAR"])

    content = dotenv_path.read_text(encoding="utf-8")
    assert "FOO.BAR=" not in content
    assert "KEEP=unchanged\n" in content


def test_update_repairs_multiline_malformed_assignment_without_dropping_following_keys(
    tmp_path,
):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "BROKEN='unterminated\nKEEP='value'\n",
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).upsert({"BROKEN": "fixed"})

    content = dotenv_path.read_text(encoding="utf-8")
    parsed = dotenv_values(dotenv_path)
    assert "KEEP='value'\n" in content
    assert _count_key_occurrences(content, "BROKEN") == 1
    assert parsed["BROKEN"] == "fixed"
    assert parsed["KEEP"] == "value"


def test_unset_multiline_malformed_assignment_preserves_following_keys(
    tmp_path,
):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "BROKEN='unterminated\nKEEP='value'\n",
        encoding="utf-8",
    )

    DotenvHandler(dotenv_path).unset(["BROKEN"])

    content = dotenv_path.read_text(encoding="utf-8")
    assert "BROKEN=" not in content
    assert "KEEP='value'\n" in content
    assert dotenv_values(dotenv_path)["KEEP"] == "value"


def test_update_applies_writes_and_removals_in_single_hardened_rewrite(
    tmp_path, monkeypatch
):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "REMOVE=old\nKEEP=unchanged\n",
        encoding="utf-8",
    )
    dotenv_path.chmod(0o644)
    atomic_writes = []
    original_atomic_write = DotenvHandler._atomic_write

    def spy_atomic_write(self, path, content):
        atomic_writes.append(content)
        return original_atomic_write(self, path, content)

    monkeypatch.setattr(DotenvHandler, "_atomic_write", spy_atomic_write)

    DotenvHandler(dotenv_path).update(
        updates={"ADD": "value with spaces"},
        removals=["REMOVE", "MISSING"],
    )

    content = dotenv_path.read_text(encoding="utf-8")
    parsed = dotenv_values(dotenv_path)

    assert len(atomic_writes) == 1
    assert "REMOVE=" not in content
    assert parsed["ADD"] == "value with spaces"
    assert parsed["KEEP"] == "unchanged"
    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600


def test_update_settings_persistence_batches_dotenv_writes_and_unsets(
    tmp_path, monkeypatch
):
    from deepeval.config.settings_manager import update_settings_and_persist

    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "GRPC_VERBOSITY=INFO\nLOG_LEVEL=20\n",
        encoding="utf-8",
    )
    atomic_writes = []
    original_atomic_write = DotenvHandler._atomic_write

    def spy_atomic_write(self, path, content):
        atomic_writes.append(content)
        return original_atomic_write(self, path, content)

    monkeypatch.setattr(DotenvHandler, "_atomic_write", spy_atomic_write)

    handled, path = update_settings_and_persist(
        {"LOG_LEVEL": 40},
        save=f"dotenv:{dotenv_path}",
        unset=["GRPC_VERBOSITY"],
    )

    parsed = dotenv_values(dotenv_path)

    assert handled is True
    assert path == dotenv_path
    assert len(atomic_writes) == 1
    assert parsed["LOG_LEVEL"] == "40"
    assert "GRPC_VERBOSITY" not in parsed


def test_unset_only_hardens_existing_dotenv_mode(tmp_path):
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        "REMOVE=old\nKEEP=unchanged\n",
        encoding="utf-8",
    )
    dotenv_path.chmod(0o644)

    DotenvHandler(dotenv_path).unset(["REMOVE"])

    content = dotenv_path.read_text(encoding="utf-8")
    assert "REMOVE=" not in content
    assert "KEEP=unchanged\n" in content
    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600


def test_settings_edit_repairs_malformed_dotenv_without_dotenv_warning(
    tmp_path, caplog, reset_settings_after_test
):
    from pydantic import SecretStr

    from deepeval.config.settings import reset_settings

    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text(
        'OPENAI_API_KEY="token with " quote"\nKEEP=unchanged\n',
        encoding="utf-8",
    )

    settings = reset_settings(reload_dotenv=False)
    with settings.edit(save=f"dotenv:{dotenv_path}"):
        settings.OPENAI_API_KEY = SecretStr('fixed " value')

    parsed = dotenv_values(dotenv_path)
    assert parsed["OPENAI_API_KEY"] == 'fixed " value'
    assert parsed["KEEP"] == "unchanged"
    assert not [
        record for record in caplog.records if record.name.startswith("dotenv")
    ]


def test_update_rejects_existing_symlinked_dotenv_path(tmp_path):
    target_path = tmp_path / "target.env"
    target_path.write_text("EXISTING=old\n", encoding="utf-8")
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv") as exc:
        DotenvHandler(link_path).upsert({"EXISTING": "updated"})

    assert isinstance(exc.value, ValueError)
    assert not isinstance(exc.value, BadParameter)
    assert "--save" not in str(exc.value)
    assert link_path.is_symlink()
    assert dotenv_values(target_path)["EXISTING"] == "old"


def test_update_rejects_symlinked_dotenv_parent_directory(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "linked"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        DotenvHandler(link_dir / ".env.local").upsert({"EXISTING": "updated"})

    assert not (real_dir / ".env.local").exists()


def test_update_rejects_symlinked_dotenv_ancestor(tmp_path):
    real_root = tmp_path / "real-root"
    real_parent = real_root / "settings"
    real_parent.mkdir(parents=True)
    link_root = tmp_path / "linked-root"
    link_root.symlink_to(real_root, target_is_directory=True)
    dotenv_path = link_root / "settings" / ".env.local"

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        DotenvHandler(dotenv_path).upsert({"EXISTING": "updated"})

    assert not (real_parent / ".env.local").exists()


def test_validate_allows_known_macos_platform_alias(monkeypatch):
    path = Path("/var/folders/deepeval/.env.local")
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def fake_is_symlink(self):
        if self == Path("/var"):
            return True
        return original_is_symlink(self)

    def fake_resolve(self, strict=False):
        if self == Path("/var"):
            return Path("/private/var")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", fake_resolve)

    assert DotenvHandler(path).validate_target() == path


def test_unset_rejects_broken_symlinked_dotenv_path(tmp_path):
    target_path = tmp_path / "missing.env"
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    assert link_path.is_symlink()
    assert not link_path.exists()

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        DotenvHandler(link_path).unset(["EXISTING"])


def test_settings_edit_rejects_symlink_before_dotenv_pre_read(
    tmp_path, monkeypatch, reset_settings_after_test
):
    import deepeval.config.settings as settings_mod
    from deepeval.config.settings import reset_settings

    settings = reset_settings(reload_dotenv=False)
    previous_log_level = settings.LOG_LEVEL
    target_path = tmp_path / "target.env"
    target_path.write_text("LOG_LEVEL=20\n", encoding="utf-8")
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    def fail_if_read(path):
        raise AssertionError("symlinked dotenv target should not be read")

    monkeypatch.setattr(settings_mod, "read_dotenv_file_silent", fail_if_read)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        with settings.edit(save=f"dotenv:{link_path}"):
            settings.LOG_LEVEL = 40

    assert settings.LOG_LEVEL == previous_log_level


def test_settings_edit_rolls_back_when_dotenv_pre_read_fails(
    tmp_path, monkeypatch, reset_settings_after_test
):
    from deepeval.config.settings import reset_settings

    settings = reset_settings(reload_dotenv=False)
    previous_log_level = settings.LOG_LEVEL
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    with pytest.raises(IsADirectoryError):
        with settings.edit(save=f"dotenv:{tmp_path}"):
            settings.LOG_LEVEL = 40

    assert settings.LOG_LEVEL == previous_log_level
    assert "LOG_LEVEL" not in os.environ


def test_update_settings_rolls_back_runtime_env_when_dotenv_persist_fails(
    tmp_path, monkeypatch, reset_settings_after_test
):
    from deepeval.config.settings import reset_settings
    from deepeval.config.settings_manager import update_settings_and_persist

    settings = reset_settings(reload_dotenv=False)
    previous_log_level = settings.LOG_LEVEL
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    target_path = tmp_path / "target.env"
    target_path.write_text("LOG_LEVEL=20\n", encoding="utf-8")
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        update_settings_and_persist(
            {"LOG_LEVEL": 40},
            save=f"dotenv:{link_path}",
        )

    assert settings.LOG_LEVEL == previous_log_level
    assert "LOG_LEVEL" not in os.environ


def test_settings_edit_rolls_back_runtime_when_dotenv_persist_fails(
    tmp_path, monkeypatch, reset_settings_after_test
):
    from deepeval.config.settings import reset_settings

    settings = reset_settings(reload_dotenv=False)
    previous_log_level = settings.LOG_LEVEL
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    target_path = tmp_path / "target.env"
    target_path.write_text("LOG_LEVEL=20\n", encoding="utf-8")
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        with settings.edit(save=f"dotenv:{link_path}"):
            settings.LOG_LEVEL = 40

    assert settings.LOG_LEVEL == previous_log_level
    assert "LOG_LEVEL" not in os.environ


def test_update_settings_rolls_back_after_atomic_write_failure(
    tmp_path, monkeypatch, reset_settings_after_test
):
    from deepeval.config.settings import reset_settings
    from deepeval.config.settings_manager import update_settings_and_persist

    settings = reset_settings(reload_dotenv=False)
    previous_log_level = settings.LOG_LEVEL
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text("LOG_LEVEL=20\n", encoding="utf-8")

    def fail_atomic_write(self, path, content):
        raise OSError("simulated dotenv write failure")

    monkeypatch.setattr(DotenvHandler, "_atomic_write", fail_atomic_write)

    with pytest.raises(OSError, match="simulated dotenv write failure"):
        update_settings_and_persist(
            {"LOG_LEVEL": 40},
            save=f"dotenv:{dotenv_path}",
        )

    assert settings.LOG_LEVEL == previous_log_level
    assert "LOG_LEVEL" not in os.environ
    assert dotenv_values(dotenv_path)["LOG_LEVEL"] == "20"


def test_settings_edit_restores_legacy_store_after_atomic_write_failure(
    tmp_path, monkeypatch, reset_settings_after_test
):
    from deepeval import key_handler as key_handler_mod
    from deepeval.config.settings import reset_settings

    hidden_store_dir = tmp_path / "hidden-store"
    hidden_store_dir.mkdir()
    keyfile_path = hidden_store_dir / key_handler_mod.KEY_FILE
    original_keyfile = {"TEMPERATURE": "0.5"}
    keyfile_path.write_text(json.dumps(original_keyfile), encoding="utf-8")
    monkeypatch.setattr(
        key_handler_mod, "HIDDEN_DIR", str(hidden_store_dir), raising=False
    )
    monkeypatch.delenv("TEMPERATURE", raising=False)

    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text("TEMPERATURE=0.5\n", encoding="utf-8")
    settings = reset_settings(reload_dotenv=False)
    previous_temperature = settings.TEMPERATURE

    def fail_atomic_write(self, path, content):
        raise OSError("simulated dotenv write failure")

    monkeypatch.setattr(DotenvHandler, "_atomic_write", fail_atomic_write)

    with pytest.raises(OSError, match="simulated dotenv write failure"):
        with settings.edit(save=f"dotenv:{dotenv_path}"):
            settings.TEMPERATURE = 0.7

    assert settings.TEMPERATURE == previous_temperature
    assert "TEMPERATURE" not in os.environ
    assert (
        json.loads(keyfile_path.read_text(encoding="utf-8")) == original_keyfile
    )
    assert dotenv_values(dotenv_path)["TEMPERATURE"] == "0.5"


def test_settings_provider_switch_rolls_back_legacy_store_on_dotenv_failure(
    tmp_path, monkeypatch, reset_settings_after_test
):
    from deepeval import key_handler as key_handler_mod
    from deepeval.config.settings import reset_settings
    from deepeval.key_handler import ModelKeyValues

    hidden_store_dir = tmp_path / "hidden-store"
    hidden_store_dir.mkdir()
    keyfile_path = hidden_store_dir / key_handler_mod.KEY_FILE
    original_keyfile = {"USE_OPENAI_MODEL": "YES"}
    keyfile_path.write_text(json.dumps(original_keyfile), encoding="utf-8")
    monkeypatch.setattr(
        key_handler_mod, "HIDDEN_DIR", str(hidden_store_dir), raising=False
    )

    settings = reset_settings(reload_dotenv=False)
    previous_openai = settings.USE_OPENAI_MODEL
    previous_local = settings.USE_LOCAL_MODEL
    target_path = tmp_path / "target.env"
    target_path.write_text("USE_OPENAI_MODEL=1\n", encoding="utf-8")
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        settings.set_model_provider(
            ModelKeyValues.USE_LOCAL_MODEL,
            save=f"dotenv:{link_path}",
        )

    assert settings.USE_OPENAI_MODEL is previous_openai
    assert settings.USE_LOCAL_MODEL is previous_local
    assert (
        json.loads(keyfile_path.read_text(encoding="utf-8")) == original_keyfile
    )
    assert dotenv_values(target_path)["USE_OPENAI_MODEL"] == "1"


def test_cli_utils_provider_switch_leaves_legacy_store_on_dotenv_failure(
    tmp_path, monkeypatch
):
    from deepeval import key_handler as key_handler_mod
    from deepeval.cli import utils as cli_utils
    from deepeval.key_handler import ModelKeyValues

    hidden_store_dir = tmp_path / "hidden-store"
    hidden_store_dir.mkdir()
    keyfile_path = hidden_store_dir / key_handler_mod.KEY_FILE
    original_keyfile = {"USE_OPENAI_MODEL": "YES"}
    keyfile_path.write_text(json.dumps(original_keyfile), encoding="utf-8")
    monkeypatch.setattr(
        key_handler_mod, "HIDDEN_DIR", str(hidden_store_dir), raising=False
    )

    target_path = tmp_path / "target.env"
    target_path.write_text("USE_OPENAI_MODEL=1\n", encoding="utf-8")
    link_path = tmp_path / ".env.local"
    link_path.symlink_to(target_path)

    with pytest.raises(DotenvTargetError, match="symlinked dotenv"):
        cli_utils.switch_model_provider(
            ModelKeyValues.USE_LOCAL_MODEL,
            save=f"dotenv:{link_path}",
        )

    assert (
        json.loads(keyfile_path.read_text(encoding="utf-8")) == original_keyfile
    )
    assert dotenv_values(target_path)["USE_OPENAI_MODEL"] == "1"


def test_cli_utils_provider_switch_restores_dotenv_when_legacy_write_fails(
    tmp_path, monkeypatch
):
    from deepeval import key_handler as key_handler_mod
    from deepeval.cli import utils as cli_utils
    from deepeval.key_handler import ModelKeyValues

    hidden_store_dir = tmp_path / "hidden-store"
    hidden_store_dir.mkdir()
    keyfile_path = hidden_store_dir / key_handler_mod.KEY_FILE
    original_keyfile = {"USE_OPENAI_MODEL": "YES"}
    keyfile_path.write_text(json.dumps(original_keyfile), encoding="utf-8")
    monkeypatch.setattr(
        key_handler_mod, "HIDDEN_DIR", str(hidden_store_dir), raising=False
    )

    dotenv_path = tmp_path / ".env.local"
    original_dotenv = "USE_OPENAI_MODEL=true\nKEEP=unchanged\n"
    dotenv_path.write_text(original_dotenv, encoding="utf-8")
    dotenv_path.chmod(0o644)

    def fail_write_key(key, value):
        raise OSError("simulated legacy write failure")

    monkeypatch.setattr(
        cli_utils.KEY_FILE_HANDLER,
        "write_key",
        fail_write_key,
    )

    with pytest.raises(OSError, match="simulated legacy write failure"):
        cli_utils.switch_model_provider(
            ModelKeyValues.USE_LOCAL_MODEL,
            save=f"dotenv:{dotenv_path}",
        )

    assert dotenv_path.read_text(encoding="utf-8") == original_dotenv
    assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o644
    assert (
        json.loads(keyfile_path.read_text(encoding="utf-8")) == original_keyfile
    )
