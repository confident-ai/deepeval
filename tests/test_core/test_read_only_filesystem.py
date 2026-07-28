import os
import subprocess
import sys
from pathlib import Path

from deepeval.config.settings import get_settings, reset_settings


def _enable_read_only(monkeypatch) -> None:
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)


def test_import_does_not_create_internal_store_in_read_only_mode(
    tmp_path: Path,
):
    subprocess_cwd = tmp_path / "subprocess"
    subprocess_cwd.mkdir()
    env = os.environ.copy()
    env["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
    env["DEEPEVAL_TELEMETRY_OPT_OUT"] = "0"

    result = subprocess.run(
        [sys.executable, "-c", "import deepeval.telemetry"],
        cwd=subprocess_cwd,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not (subprocess_cwd / ".deepeval").exists()


def test_settings_edit_updates_runtime_without_persisting_dotenv(
    tmp_path: Path, monkeypatch
):
    _enable_read_only(monkeypatch)
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text("EXISTING=value\n", encoding="utf-8")

    settings = get_settings()
    with settings.edit(save=f"dotenv:{dotenv_path}"):
        settings.GRPC_VERBOSITY = "ERROR"

    assert os.environ["GRPC_VERBOSITY"] == "ERROR"
    assert dotenv_path.read_text(encoding="utf-8") == "EXISTING=value\n"


def test_legacy_keystore_is_not_mutated_in_read_only_mode(
    tmp_path: Path, monkeypatch
):
    import deepeval.key_handler as key_handler

    _enable_read_only(monkeypatch)
    hidden_dir = tmp_path / ".deepeval"
    key_file = hidden_dir / key_handler.KEY_FILE
    hidden_dir.mkdir(exist_ok=True)
    key_file.write_text(
        '{"confident_region": "US"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(key_handler, "HIDDEN_DIR", str(hidden_dir))

    handler = key_handler.KeyFileHandler()
    handler.write_key(key_handler.KeyValues.CONFIDENT_REGION, "EU")
    handler.remove_key(key_handler.KeyValues.CONFIDENT_REGION)

    assert key_file.read_text(encoding="utf-8") == (
        '{"confident_region": "US"}'
    )


def test_telemetry_is_not_persisted_in_read_only_mode(
    tmp_path: Path, monkeypatch
):
    import deepeval.telemetry as telemetry

    _enable_read_only(monkeypatch)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "0")
    reset_settings(reload_dotenv=False)
    telemetry_path = tmp_path / ".deepeval" / telemetry.TELEMETRY_DATA_FILE
    monkeypatch.setattr(telemetry, "HIDDEN_DIR", str(telemetry_path.parent))
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", str(telemetry_path))

    unique_id = telemetry.get_unique_id()
    telemetry.set_last_feature(telemetry.Feature.EVALUATION)

    assert unique_id
    assert not telemetry_path.exists()


def test_runtime_cache_writes_stop_after_switching_to_read_only(
    tmp_path: Path, monkeypatch
):
    import deepeval.prompt.prompt as prompt_module
    import deepeval.test_run.cache as cache_module

    _enable_read_only(monkeypatch)

    prompt_cache = tmp_path / "prompt-cache.json"
    monkeypatch.setattr(
        prompt_module,
        "CACHE_FILE_NAME",
        str(prompt_cache),
    )
    prompt = prompt_module.Prompt(alias="read-only-test")
    prompt._write_to_cache(
        prompt_module.HASH_CACHE_KEY,
        hash="abc123",
    )

    test_run_cache = tmp_path / "test-run-cache.json"
    manager = cache_module.TestRunCacheManager()
    manager.cache_file_name = str(test_run_cache)
    manager.create_cached_test_run()

    assert not prompt_cache.exists()
    assert not test_run_cache.exists()


def test_internal_cleanup_does_not_delete_files_in_read_only_mode(
    tmp_path: Path, monkeypatch
):
    from deepeval.utils import delete_file_if_exists

    _enable_read_only(monkeypatch)
    internal_file = tmp_path / ".temp_test_run_data.json"
    internal_file.write_text("keep", encoding="utf-8")

    delete_file_if_exists(internal_file)

    assert internal_file.read_text(encoding="utf-8") == "keep"
