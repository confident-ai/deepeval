import asyncio
import importlib
import json
import os
import portalocker

import pytest

import deepeval.cli.utils as cli_utils

import deepeval.test_run.cache as cache_mod
import deepeval.test_run.test_run as tr_mod

from pathlib import Path
from types import SimpleNamespace

from deepeval.config.settings import reset_settings
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRun, TestRunManager, LLMApiTestCase
from tests.test_core.helpers import _make_fake_portalocker
from tests.test_core.stubs import RecordingPortalockerLock

eval_e2e_mod = importlib.import_module("deepeval.evaluate.execute.e2e")
eval_loop_mod = importlib.import_module("deepeval.evaluate.execute.loop")
evaluate_mod = importlib.import_module("deepeval.evaluate.evaluate")
execute_common_mod = importlib.import_module(
    "deepeval.evaluate.execute._common"
)
trace_scope_mod = importlib.import_module(
    "deepeval.evaluate.execute.trace_scope"
)
hyperparameters_mod = importlib.import_module(
    "deepeval.test_run.hyperparameters"
)
test_run_pkg_mod = importlib.import_module("deepeval.test_run")
cli_inspect_mod = importlib.import_module("deepeval.cli.inspect")


class _AlwaysPassMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.5
        self.strict_mode = False
        self.score = None
        self.reason = None
        self.success = False
        self.error = None
        self.evaluation_model = None
        self.evaluation_cost = None
        self.input_tokens = None
        self.output_tokens = None
        self.verbose_logs = None

    @property
    def __name__(self):
        return "AlwaysPass"

    def measure(self, test_case, *args, **kwargs):
        self.score = 1.0
        self.success = True
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success


def _reset_test_run_singletons():
    tr_mod.global_test_run_manager.reset()
    tr_mod.global_test_run_cache_manager.disable_write_cache = None
    tr_mod.global_test_run_cache_manager.cached_test_run = None
    tr_mod.global_test_run_cache_manager.temp_cached_test_run = None
    tr_mod.global_test_run_cache_manager._temp_cache_session_started = False


@pytest.fixture
def isolated_test_run_singletons():
    _reset_test_run_singletons()
    yield
    _reset_test_run_singletons()


@pytest.fixture(autouse=True)
def isolated_hidden_artifact_paths(monkeypatch, tmp_path):
    hidden_dir = tmp_path / ".deepeval"
    temp_file = hidden_dir / ".temp_test_run_data.json"
    latest_file = hidden_dir / ".latest_test_run.json"
    latest_full_file = hidden_dir / ".latest_run_full.json"
    cache_file = hidden_dir / ".deepeval-cache.json"
    temp_cache_file = hidden_dir / ".temp-deepeval-cache.json"

    # Several modules import these path constants by value. Patch each import
    # site used by this test file so filesystem assertions stay tmp_path-local.
    monkeypatch.setattr(tr_mod, "TEMP_FILE_PATH", str(temp_file))
    monkeypatch.setattr(test_run_pkg_mod, "TEMP_FILE_PATH", str(temp_file))
    monkeypatch.setattr(evaluate_mod, "TEMP_FILE_PATH", str(temp_file))
    monkeypatch.setattr(execute_common_mod, "TEMP_FILE_PATH", str(temp_file))
    monkeypatch.setattr(trace_scope_mod, "TEMP_FILE_PATH", str(temp_file))
    monkeypatch.setattr(hyperparameters_mod, "TEMP_FILE_PATH", str(temp_file))
    monkeypatch.setattr(tr_mod, "LATEST_TEST_RUN_FILE_PATH", str(latest_file))
    monkeypatch.setattr(
        test_run_pkg_mod, "LATEST_TEST_RUN_FILE_PATH", str(latest_file)
    )
    monkeypatch.setattr(
        tr_mod, "LATEST_FULL_TEST_RUN_FILE_PATH", str(latest_full_file)
    )
    monkeypatch.setattr(cache_mod, "CACHE_FILE_NAME", str(cache_file))
    monkeypatch.setattr(cache_mod, "TEMP_CACHE_FILE_NAME", str(temp_cache_file))
    monkeypatch.setattr(
        tr_mod.global_test_run_manager, "temp_file_path", str(temp_file)
    )
    monkeypatch.setattr(
        tr_mod.global_test_run_cache_manager,
        "cache_file_name",
        str(cache_file),
    )
    monkeypatch.setattr(
        tr_mod.global_test_run_cache_manager,
        "temp_cache_file_name",
        str(temp_cache_file),
    )


def _implicit_test_run_payload_paths():
    return [
        Path(tr_mod.TEMP_FILE_PATH),
        Path(tr_mod.LATEST_TEST_RUN_FILE_PATH),
        Path(tr_mod.LATEST_FULL_TEST_RUN_FILE_PATH),
    ]


def _cache_artifact_paths():
    return [
        Path(cache_mod.CACHE_FILE_NAME),
        Path(cache_mod.TEMP_CACHE_FILE_NAME),
    ]


def _assert_no_implicit_test_run_payloads():
    for path in _implicit_test_run_payload_paths():
        assert not path.exists(), f"{path} should not exist"


def _assert_no_cache_artifacts():
    for path in _cache_artifact_paths():
        assert not path.exists(), f"{path} should not exist"


def _write_stale_hidden_artifacts():
    for path, content in [
        (Path(tr_mod.TEMP_FILE_PATH), "stale temp run"),
        (Path(tr_mod.LATEST_TEST_RUN_FILE_PATH), "stale latest run"),
        (Path(tr_mod.LATEST_FULL_TEST_RUN_FILE_PATH), "stale rolling snapshot"),
        (Path(cache_mod.CACHE_FILE_NAME), "stale cache"),
        (Path(cache_mod.TEMP_CACHE_FILE_NAME), "stale temp cache"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _assert_stale_hidden_artifacts_handled_after_no_cache():
    assert (
        Path(tr_mod.TEMP_FILE_PATH).read_text(encoding="utf-8")
        == "stale temp run"
    )
    assert (
        Path(tr_mod.LATEST_TEST_RUN_FILE_PATH).read_text(encoding="utf-8")
        == "stale latest run"
    )
    assert (
        Path(tr_mod.LATEST_FULL_TEST_RUN_FILE_PATH).read_text(encoding="utf-8")
        == "stale rolling snapshot"
    )
    assert (
        Path(cache_mod.CACHE_FILE_NAME).read_text(encoding="utf-8")
        == "stale cache"
    )
    assert (
        Path(cache_mod.TEMP_CACHE_FILE_NAME).read_text(encoding="utf-8")
        == "stale temp cache"
    )


def _latest_test_run_wrapper(identifier="stale-hidden", owner_token=None):
    payload = {
        tr_mod.LATEST_TEST_RUN_DATA_KEY: TestRun(
            identifier=identifier
        ).model_dump(by_alias=True, exclude_none=True)
    }
    if owner_token is not None:
        payload[tr_mod.LATEST_TEST_RUN_OWNER_TOKEN_KEY] = owner_token
    return payload


def _run_no_cache_evaluate(monkeypatch, **display_kwargs):
    monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
    monkeypatch.setattr(tr_mod.console, "print", lambda *args, **kwargs: None)
    return evaluate(
        test_cases=[LLMTestCase(input="in", actual_output="out")],
        metrics=[_AlwaysPassMetric()],
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
            **display_kwargs,
        ),
    )


def _run_quiet_evaluate(monkeypatch, *, cache_config=None, **display_kwargs):
    monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
    monkeypatch.setattr(tr_mod.console, "print", lambda *args, **kwargs: None)
    return evaluate(
        test_cases=[LLMTestCase(input="in", actual_output="out")],
        metrics=[_AlwaysPassMetric()],
        async_config=AsyncConfig(run_async=False),
        cache_config=cache_config
        or CacheConfig(write_cache=True, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
            **display_kwargs,
        ),
    )


def _assert_exported_test_run(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["testPassed"] == 1
    assert payload["testFailed"] == 0
    test_cases = payload["testCases"]
    assert len(test_cases) == 1
    assert test_cases[0]["input"] == "in"
    assert test_cases[0]["actualOutput"] == "out"
    assert test_cases[0]["success"] is True


def test_evaluate_write_cache_false_skips_implicit_hidden_writes(
    isolated_test_run_singletons, monkeypatch
):
    _run_no_cache_evaluate(monkeypatch)

    _assert_no_implicit_test_run_payloads()
    _assert_no_cache_artifacts()
    assert tr_mod.global_test_run_manager.last_saved_path is None


def test_evaluate_default_async_write_cache_false_skips_hidden_writes(
    isolated_test_run_singletons, monkeypatch
):
    monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
    monkeypatch.setattr(tr_mod.console, "print", lambda *args, **kwargs: None)

    evaluate(
        test_cases=[LLMTestCase(input="in", actual_output="out")],
        metrics=[_AlwaysPassMetric()],
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
    )

    _assert_no_implicit_test_run_payloads()
    _assert_no_cache_artifacts()
    assert tr_mod.global_test_run_manager.last_saved_path is None


def test_evaluate_write_cache_false_invalidates_stale_latest_files(
    isolated_test_run_singletons, monkeypatch
):
    _write_stale_hidden_artifacts()

    _run_no_cache_evaluate(monkeypatch)

    _assert_stale_hidden_artifacts_handled_after_no_cache()
    assert tr_mod.global_test_run_manager.get_latest_test_run_data() is None
    assert TestRunManager().get_latest_test_run_data() is None


def test_evaluate_write_cache_true_does_not_leave_temp_metric_cache(
    isolated_test_run_singletons, monkeypatch
):
    monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
    monkeypatch.setattr(tr_mod.console, "print", lambda *args, **kwargs: None)

    evaluate(
        test_cases=[LLMTestCase(input="in", actual_output="out")],
        metrics=[_AlwaysPassMetric()],
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=True, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
    )

    assert not Path(cache_mod.TEMP_CACHE_FILE_NAME).exists()


def test_evaluate_successful_write_cache_keeps_latest_run_readable(
    isolated_test_run_singletons, monkeypatch
):
    tr_mod.global_test_run_manager._latest_test_run_cache_valid = False

    _run_quiet_evaluate(
        monkeypatch,
        cache_config=CacheConfig(write_cache=True, use_cache=False),
    )

    latest_run = tr_mod.global_test_run_manager.get_latest_test_run_data()
    assert latest_run is not None
    assert latest_run.test_passed == 1
    assert latest_run.test_failed == 0


def test_evaluate_write_cache_false_still_allows_explicit_results_export(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    results_folder = tmp_path / "exports"

    _run_no_cache_evaluate(monkeypatch, results_folder=str(results_folder))

    exported_files = list(results_folder.glob("test_run_*.json"))
    assert len(exported_files) == 1
    assert tr_mod.global_test_run_manager.last_saved_path == exported_files[0]
    _assert_exported_test_run(exported_files[0])
    _assert_no_implicit_test_run_payloads()
    _assert_no_cache_artifacts()


def test_evaluate_read_only_allows_explicit_results_export(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    results_folder = tmp_path / "read-only-exports"
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)

    try:
        _run_quiet_evaluate(
            monkeypatch,
            cache_config=CacheConfig(write_cache=True, use_cache=False),
            results_folder=str(results_folder),
        )
    finally:
        monkeypatch.delenv("DEEPEVAL_FILE_SYSTEM", raising=False)
        reset_settings(reload_dotenv=False)

    exported_files = list(results_folder.glob("test_run_*.json"))
    assert len(exported_files) == 1
    assert tr_mod.global_test_run_manager.last_saved_path == exported_files[0]
    _assert_exported_test_run(exported_files[0])
    _assert_no_implicit_test_run_payloads()
    _assert_no_cache_artifacts()


def test_evaluate_read_only_allows_explicit_console_report_export(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    output_dir = tmp_path / "console-report"
    export_calls = []
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)

    try:
        monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
        monkeypatch.setattr(
            tr_mod.console, "print", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            EvaluationConsoleReport,
            "export_to_markdown",
            lambda *args, **kwargs: export_calls.append((args, kwargs)),
        )
        evaluate(
            test_cases=[LLMTestCase(input="in", actual_output="out")],
            metrics=[_AlwaysPassMetric()],
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=True, use_cache=False),
            display_config=DisplayConfig(
                show_indicator=False,
                print_results=True,
                inspect_after_run=False,
                file_output_dir=str(output_dir),
                file_type="md",
            ),
        )
    finally:
        monkeypatch.delenv("DEEPEVAL_FILE_SYSTEM", raising=False)
        reset_settings(reload_dotenv=False)

    assert len(export_calls) == 1
    assert export_calls[0][1]["output_dir"] == str(output_dir)
    _assert_no_implicit_test_run_payloads()
    _assert_no_cache_artifacts()


def test_dataset_read_only_allows_explicit_console_report_export(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    output_dir = tmp_path / "dataset-console-report"
    export_calls = []
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)

    try:
        monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
        monkeypatch.setattr(
            tr_mod.console, "print", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            EvaluationConsoleReport,
            "export_to_markdown",
            lambda *args, **kwargs: export_calls.append((args, kwargs)),
        )

        dataset = EvaluationDataset(goldens=[Golden(input="in")])
        for golden in dataset.evals_iterator(
            metrics=[_AlwaysPassMetric()],
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=True, use_cache=False),
            display_config=DisplayConfig(
                show_indicator=False,
                print_results=True,
                inspect_after_run=False,
                file_output_dir=str(output_dir),
                file_type="md",
            ),
        ):
            assert golden.input == "in"
    finally:
        monkeypatch.delenv("DEEPEVAL_FILE_SYSTEM", raising=False)
        reset_settings(reload_dotenv=False)

    assert len(export_calls) == 1
    assert export_calls[0][1]["output_dir"] == str(output_dir)
    _assert_no_implicit_test_run_payloads()
    _assert_no_cache_artifacts()


def test_dataset_evals_iterator_restores_cache_policy_on_success(
    isolated_test_run_singletons, monkeypatch
):
    tr_mod.global_test_run_manager.save_to_disk = True
    tr_mod.global_test_run_cache_manager.disable_write_cache = False
    monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
    monkeypatch.setattr(tr_mod.console, "print", lambda *args, **kwargs: None)

    dataset = EvaluationDataset(goldens=[Golden(input="in")])
    for golden in dataset.evals_iterator(
        metrics=[_AlwaysPassMetric()],
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
    ):
        assert golden.input == "in"

    assert tr_mod.global_test_run_manager.save_to_disk is True
    assert tr_mod.global_test_run_cache_manager.disable_write_cache is False


def test_no_cache_run_does_not_upload_stale_latest_in_current_process(
    isolated_test_run_singletons, monkeypatch
):
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper()),
        encoding="utf-8",
    )

    _run_no_cache_evaluate(monkeypatch)

    post_calls = []
    monkeypatch.setattr(
        tr_mod.global_test_run_manager,
        "post_test_run",
        lambda *args, **kwargs: post_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(cli_utils, "print", lambda *args, **kwargs: None)

    cli_utils.upload_and_open_link()

    assert latest_path.exists()
    assert tr_mod.global_test_run_manager.get_latest_test_run_data() is None
    assert TestRunManager().get_latest_test_run_data() is None
    assert post_calls == []


def test_read_only_fresh_manager_ignores_stale_latest_run(
    isolated_test_run_singletons, monkeypatch
):
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper()),
        encoding="utf-8",
    )

    fresh_manager = TestRunManager()
    post_calls = []
    monkeypatch.setattr(
        fresh_manager,
        "post_test_run",
        lambda *args, **kwargs: post_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(cli_utils, "global_test_run_manager", fresh_manager)
    monkeypatch.setattr(cli_utils, "print", lambda *args, **kwargs: None)
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)

    try:
        assert fresh_manager.get_latest_test_run_data() is None
        cli_utils.upload_and_open_link()
    finally:
        monkeypatch.delenv("DEEPEVAL_FILE_SYSTEM", raising=False)
        reset_settings(reload_dotenv=False)

    assert post_calls == []


def test_read_only_configure_write_cache_false_does_not_read_hidden_latest(
    isolated_test_run_singletons, monkeypatch
):
    def fail_if_latest_file_is_read(file_path):
        raise AssertionError(f"unexpected hidden latest read: {file_path}")

    manager = TestRunManager()
    monkeypatch.setattr(
        tr_mod, "_read_latest_file_owner_token", fail_if_latest_file_is_read
    )
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)

    try:
        manager.configure_write_cache(False)
    finally:
        monkeypatch.delenv("DEEPEVAL_FILE_SYSTEM", raising=False)
        reset_settings(reload_dotenv=False)

    assert manager._latest_test_run_snapshot_owner_tokens == {}
    assert manager._latest_test_run_cache_valid is False
    assert manager._temp_test_run_cache_valid is False


def test_read_only_inspect_default_ignores_stale_rolling_snapshot(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    rolling_path = Path(tr_mod.LATEST_FULL_TEST_RUN_FILE_PATH)
    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_path.write_text("stale rolling snapshot", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPEVAL_RESULTS_FOLDER", raising=False)
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    reset_settings(reload_dotenv=False)

    try:
        assert cli_inspect_mod._resolve_target(None, None) is None
    finally:
        monkeypatch.delenv("DEEPEVAL_FILE_SYSTEM", raising=False)
        reset_settings(reload_dotenv=False)


def test_inspect_default_ignores_unowned_rolling_snapshot(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    rolling_path = Path(tr_mod.LATEST_FULL_TEST_RUN_FILE_PATH)
    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_path.write_text("stale rolling snapshot", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPEVAL_RESULTS_FOLDER", raising=False)

    assert cli_inspect_mod._resolve_target(None, None) is None


def test_latest_reader_ignores_unowned_stale_file(
    isolated_test_run_singletons,
):
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper()),
        encoding="utf-8",
    )

    assert TestRunManager().get_latest_test_run_data() is None


def test_no_cache_latest_invalidation_deletes_same_owner_snapshot(
    isolated_test_run_singletons,
):
    manager = TestRunManager()
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper(owner_token="owned-token")),
        encoding="utf-8",
    )

    manager.configure_write_cache(False)
    manager._invalidate_latest_test_run_cache(delete_owned_disk_state=True)

    assert not latest_path.exists()


def test_no_cache_latest_invalidation_tombstones_when_delete_fails(
    isolated_test_run_singletons, monkeypatch
):
    manager = TestRunManager()
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper(owner_token="owned-token")),
        encoding="utf-8",
    )
    real_unlink = os.unlink

    def raise_permission_error(path):
        if path == str(latest_path):
            raise PermissionError("simulated locked file")
        real_unlink(path)

    monkeypatch.setattr(tr_mod.os, "unlink", raise_permission_error)

    manager.configure_write_cache(False)
    manager._invalidate_latest_test_run_cache(delete_owned_disk_state=True)

    assert latest_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert tr_mod.LATEST_TEST_RUN_OWNER_TOKEN_KEY not in payload
    assert TestRunManager().get_latest_test_run_data() is None


def test_no_cache_latest_invalidation_preserves_new_owner_replacement(
    isolated_test_run_singletons,
):
    manager = TestRunManager()
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper(owner_token="old-token")),
        encoding="utf-8",
    )

    manager.configure_write_cache(False)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper(owner_token="new-token")),
        encoding="utf-8",
    )
    manager._invalidate_latest_test_run_cache(delete_owned_disk_state=True)

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload[tr_mod.LATEST_TEST_RUN_OWNER_TOKEN_KEY] == "new-token"


def test_execute_test_cases_restores_cache_policy_on_base_exception(
    isolated_test_run_singletons, monkeypatch
):
    tr_mod.global_test_run_manager.save_to_disk = True
    tr_mod.global_test_run_cache_manager.disable_write_cache = False

    def raise_keyboard_interrupt(*args, **kwargs):
        raise KeyboardInterrupt("interrupt metric")

    monkeypatch.setattr(
        eval_e2e_mod, "_execute_metric", raise_keyboard_interrupt
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt metric"):
        eval_e2e_mod.execute_test_cases(
            test_cases=[LLMTestCase(input="in", actual_output="out")],
            metrics=[_AlwaysPassMetric()],
            error_config=ErrorConfig(ignore_errors=False),
            display_config=DisplayConfig(show_indicator=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )

    assert tr_mod.global_test_run_manager.save_to_disk is True
    assert tr_mod.global_test_run_cache_manager.disable_write_cache is False


def test_evaluate_restores_cache_policy_after_post_execution_error(
    isolated_test_run_singletons, monkeypatch, tmp_path
):
    tr_mod.global_test_run_manager.save_to_disk = True
    tr_mod.global_test_run_cache_manager.disable_write_cache = False
    latest_path = Path(tr_mod.LATEST_TEST_RUN_FILE_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(_latest_test_run_wrapper(owner_token="owned-token")),
        encoding="utf-8",
    )
    monkeypatch.setattr(tr_mod, "is_confident", lambda: False)
    monkeypatch.setattr(tr_mod.console, "print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Invalid file type"):
        evaluate(
            test_cases=[LLMTestCase(input="in", actual_output="out")],
            metrics=[_AlwaysPassMetric()],
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            display_config=DisplayConfig(
                show_indicator=False,
                print_results=True,
                inspect_after_run=False,
                file_output_dir=str(tmp_path / "console-report"),
                file_type=None,
            ),
        )

    assert tr_mod.global_test_run_manager.save_to_disk is True
    assert tr_mod.global_test_run_cache_manager.disable_write_cache is False
    assert tr_mod.global_test_run_manager.get_latest_test_run_data() is None
    assert TestRunManager().get_latest_test_run_data() is None
    assert not latest_path.exists()


def test_async_execute_test_cases_restores_cache_policy_on_cancel(
    isolated_test_run_singletons, monkeypatch
):
    tr_mod.global_test_run_manager.save_to_disk = True
    tr_mod.global_test_run_cache_manager.disable_write_cache = False

    async def scenario():
        started = asyncio.Event()
        blocker = asyncio.Event()

        async def wait_forever(*args, **kwargs):
            started.set()
            await blocker.wait()

        monkeypatch.setattr(
            eval_e2e_mod, "_a_execute_llm_test_cases", wait_forever
        )

        task = asyncio.create_task(
            eval_e2e_mod.a_execute_test_cases(
                test_cases=[LLMTestCase(input="in", actual_output="out")],
                metrics=[_AlwaysPassMetric()],
                error_config=ErrorConfig(ignore_errors=False),
                display_config=DisplayConfig(show_indicator=False),
                cache_config=CacheConfig(write_cache=False, use_cache=False),
                async_config=AsyncConfig(run_async=True),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert tr_mod.global_test_run_manager.save_to_disk is True
    assert tr_mod.global_test_run_cache_manager.disable_write_cache is False


def test_temp_cache_session_preserves_existing_temp_cache_entries(
    isolated_test_run_singletons,
):
    first = cache_mod.TestRunCacheManager()
    second = cache_mod.TestRunCacheManager()
    for manager in (first, second):
        manager.cache_file_name = cache_mod.CACHE_FILE_NAME
        manager.temp_cache_file_name = cache_mod.TEMP_CACHE_FILE_NAME

    first.cache_test_case(
        LLMTestCase(input="first", actual_output="out"),
        cache_mod.CachedTestCase(),
        hyperparameters=None,
        to_temp=True,
    )
    second.cache_test_case(
        LLMTestCase(input="second", actual_output="out"),
        cache_mod.CachedTestCase(),
        hyperparameters=None,
        to_temp=True,
    )

    payload = json.loads(
        Path(cache_mod.TEMP_CACHE_FILE_NAME).read_text(encoding="utf-8")
    )

    assert len(payload["test_cases_lookup_map"]) == 2


def test_temp_cache_wrap_up_merges_existing_temp_entries(
    isolated_test_run_singletons,
):
    first = cache_mod.TestRunCacheManager()
    second = cache_mod.TestRunCacheManager()
    for manager in (first, second):
        manager.cache_file_name = cache_mod.CACHE_FILE_NAME
        manager.temp_cache_file_name = cache_mod.TEMP_CACHE_FILE_NAME

    first.cache_test_case(
        LLMTestCase(input="first", actual_output="out"),
        cache_mod.CachedTestCase(),
        hyperparameters=None,
        to_temp=True,
    )
    second.cache_test_case(
        LLMTestCase(input="second", actual_output="out"),
        cache_mod.CachedTestCase(),
        hyperparameters=None,
        to_temp=True,
    )

    first.wrap_up_cached_test_run()

    payload = json.loads(
        Path(cache_mod.CACHE_FILE_NAME).read_text(encoding="utf-8")
    )
    assert len(payload["test_cases_lookup_map"]) == 2
    assert not Path(cache_mod.TEMP_CACHE_FILE_NAME).exists()
    assert first._temp_cache_session_started is False


def test_agentic_loop_restores_cache_policy_on_early_close(
    isolated_test_run_singletons,
):
    tr_mod.global_test_run_manager.save_to_disk = True
    tr_mod.global_test_run_cache_manager.disable_write_cache = False

    iterator = eval_loop_mod.execute_agentic_test_cases_from_loop(
        goldens=[Golden(input="in")],
        trace_metrics=[_AlwaysPassMetric()],
        test_results=[],
        display_config=DisplayConfig(show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        error_config=ErrorConfig(ignore_errors=False),
    )

    assert next(iterator).input == "in"
    iterator.close()

    assert tr_mod.global_test_run_manager.save_to_disk is True
    assert tr_mod.global_test_run_cache_manager.disable_write_cache is False


def test_async_agentic_loop_cancels_created_tasks_on_early_close(
    isolated_test_run_singletons,
):
    tr_mod.global_test_run_manager.save_to_disk = True
    tr_mod.global_test_run_cache_manager.disable_write_cache = False
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        blocker = asyncio.Event()
        iterator = eval_loop_mod.a_execute_agentic_test_cases_from_loop(
            goldens=[Golden(input="in")],
            trace_metrics=[_AlwaysPassMetric()],
            test_results=[],
            loop=loop,
            display_config=DisplayConfig(show_indicator=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            error_config=ErrorConfig(ignore_errors=False),
            async_config=AsyncConfig(run_async=True),
        )

        assert next(iterator).input == "in"
        created_task = asyncio.create_task(blocker.wait())
        loop.run_until_complete(asyncio.sleep(0))
        iterator.close()

        assert created_task.cancelled()
        assert tr_mod.global_test_run_manager.save_to_disk is True
        assert tr_mod.global_test_run_cache_manager.disable_write_cache is False
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.close()
        asyncio.set_event_loop(None)


def test_save_final_link_can_skip_hidden_persistence(
    isolated_test_run_singletons,
):
    trm = TestRunManager()

    trm.save_final_test_run_link(
        "https://example.com/test-runs/123", persist_link=False
    )

    assert not Path(tr_mod.LATEST_TEST_RUN_FILE_PATH).exists()


def test_get_test_run_preserves_valid_instance_on_read_lock(tmp_path):
    p = tmp_path / "temp_test_run_data.json"
    p.write_text("{}")

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    trm.create_test_run(identifier="repro-2243")

    # Now simulate a read lock so get_test_run() hits LockException
    lock = portalocker.Lock(
        str(p), mode="w", flags=portalocker.LOCK_EX | portalocker.LOCK_NB
    )
    lock.acquire()
    try:
        out = trm.get_test_run(identifier="repro-2243")
        assert out is not None
    finally:
        lock.release()


def test_get_test_run_preserves_instance_when_file_missing(
    tmp_path, monkeypatch
):
    p = tmp_path / "missing.json"

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    trm.create_test_run(identifier="first-run")
    # simulate file vanished between create and read
    if os.path.exists(p):
        os.remove(p)

    out = trm.get_test_run(identifier="first-run")
    assert out is not None  # preserves in-memory object


def test_get_test_run_preserves_instance_on_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json]")

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    trm.create_test_run(identifier="bad-json")

    out = trm.get_test_run(identifier="bad-json")
    assert out is not None


def test_update_test_run_falls_back_in_memory_on_read_failure(tmp_path):
    p = tmp_path / "run.json"

    trm = TestRunManager()
    trm.save_to_disk = True
    trm.temp_file_path = str(p)

    # create a valid run and write it to disk once
    trm.create_test_run(identifier="fallback")

    # corrupt the file so the subsequent read in update_test_run() JSON-decodes and fails
    p.write_text("{not valid json]")

    api_tc = LLMApiTestCase(
        name="t1",
        input="in",
        actual_output="out",
        order=0,
        metrics_data=[],
        trace=None,
    )
    llm_tc = LLMTestCase(input="in", actual_output="out")

    # this should hit the except branch and fall back to in-memory update
    trm.update_test_run(api_tc, llm_tc)

    out = trm.get_test_run()
    assert out is not None
    assert any(tc.name == "t1" for tc in out.test_cases)


def test_save_test_run_with_save_under_key_flushes_and_syncs(
    monkeypatch, tmp_path
):
    """
    When save_under_key is used, TestRunManager.save_test_run calls json.dump
    directly. We want to ensure that path flushes and fsyncs the file before releasing
    the portalocker lock.
    """
    # Patch portalocker inside the module under test
    monkeypatch.setattr(
        tr_mod, "portalocker", _make_fake_portalocker(), raising=False
    )

    # Track fsync calls
    fsync_calls: list[int] = []

    def fake_fsync(fd: int) -> None:
        fsync_calls.append(fd)

    monkeypatch.setattr(tr_mod.os, "fsync", fake_fsync)

    # Minimal "test_run" stub: only needs model_dump/dict for this path
    dummy_test_run = SimpleNamespace(
        model_dump=lambda **kwargs: {"foo": "bar"},
        dict=lambda **kwargs: {"foo": "bar"},
        save=lambda f: None,
    )

    # Minimal "self" stub: save_to_disk + test_run
    dummy_manager = SimpleNamespace(
        save_to_disk=True,
        test_run=dummy_test_run,
        _can_use_disk=lambda: True,
    )

    path = tmp_path / "run.json"

    # Call the real implementation as an unbound method
    TestRunManager.save_test_run(
        dummy_manager,
        str(path),
        save_under_key="wrapped_key",
    )

    f = RecordingPortalockerLock.last_file
    assert f is not None, "RecordingPortalockerLock did not capture a file"

    assert f.flushed, (
        "save_test_run(..., save_under_key=...) should call file.flush() "
        "after json.dump(...)"
    )
    assert (
        fsync_calls
    ), "save_test_run(..., save_under_key=...) should call os.fsync(file.fileno())"
    assert fsync_calls[-1] == f.fileno()
