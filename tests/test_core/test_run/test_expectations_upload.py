from unittest.mock import Mock

import pytest

import deepeval.test_run.test_run as tr_mod
from deepeval.dataset import Expectations
from deepeval.test_run.api import LLMApiTestCase, ConversationalApiTestCase


def make_run(kind, empty=False):
    expectations = Expectations() if empty else Expectations(must=["Answer"])
    run = tr_mod.TestRun()
    # Check the whole run, including cases beyond the first upload batch.
    run.test_cases = [
        LLMApiTestCase(name=str(i), input="Hi", success=True) for i in range(41)
    ]
    if kind == "single":
        run.test_cases[-1].expectations = expectations
    else:
        run.conversational_test_cases = [
            ConversationalApiTestCase(
                name="conversation",
                success=True,
                metricsData=[],
                expectations=expectations,
            )
        ]
    return run


@pytest.mark.parametrize("kind", ["single", "conversation"])
@pytest.mark.parametrize("empty", [False, True])
def test_post_blocks_expectations_before_api_creation(
    monkeypatch, capsys, kind, empty
):
    run = make_run(kind, empty)
    api = Mock(side_effect=AssertionError("Must not create an API client"))
    monkeypatch.setattr(tr_mod, "Api", api)

    assert tr_mod.TestRunManager().post_test_run(run) is None

    api.assert_not_called()
    assert len(run.test_cases) == 41
    assert "not available on Confident AI yet" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["single", "conversation"])
@pytest.mark.parametrize("logged_in", [False, True])
def test_wrap_up_expectations_uses_local_results(
    monkeypatch, capsys, kind, logged_in
):
    manager = tr_mod.TestRunManager()
    manager.disable_request = False
    run = make_run(kind)
    monkeypatch.setattr(manager, "get_test_run", lambda: run)
    monkeypatch.setattr(tr_mod, "is_confident", lambda: logged_in)
    monkeypatch.setattr(tr_mod, "delete_file_if_exists", Mock())
    monkeypatch.setattr(
        tr_mod.global_test_run_cache_manager, "wrap_up_cached_test_run", Mock()
    )
    for name in (
        "save_test_run_locally",
        "save_test_run",
        "post_test_run",
        "display_results_table",
    ):
        monkeypatch.setattr(manager, name, Mock())
    telemetry = Mock()
    monkeypatch.setattr(tr_mod, "capture_login_prompt_shown", telemetry)

    assert manager.wrap_up_test_run(1.0) is None

    manager.post_test_run.assert_not_called()
    manager.save_test_run_locally.assert_called_once()
    manager.save_test_run.assert_called_once()
    manager.display_results_table.assert_called_once()
    telemetry.assert_not_called()
    output = capsys.readouterr().out
    assert "not available on Confident AI yet" in output
    assert "Evaluation completed" in output
    assert "Pass Rate:" in output
    assert "deepeval view" not in output
    assert "Posting the run anyway" not in output


def test_wrap_up_without_expectations_still_uploads(monkeypatch):
    manager = tr_mod.TestRunManager()
    manager.disable_request = False
    run = tr_mod.TestRun(
        testCases=[LLMApiTestCase(name="case", input="Hi", success=True)]
    )
    monkeypatch.setattr(manager, "get_test_run", lambda: run)
    monkeypatch.setattr(tr_mod, "is_confident", lambda: True)
    monkeypatch.setattr(tr_mod, "delete_file_if_exists", Mock())
    monkeypatch.setattr(
        tr_mod.global_test_run_cache_manager, "wrap_up_cached_test_run", Mock()
    )
    monkeypatch.setattr(manager, "save_test_run_locally", Mock())
    monkeypatch.setattr(manager, "_record_confident_test_run_id", Mock())
    post = Mock(return_value=("link", "run-id"))
    monkeypatch.setattr(manager, "post_test_run", post)

    assert manager.wrap_up_test_run(1.0, display_table=False) == (
        "link",
        "run-id",
    )
    post.assert_called_once_with(run)
