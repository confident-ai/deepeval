from unittest.mock import Mock

import pytest

import deepeval.test_run.test_run as tr_mod
from deepeval.dataset import Expectations
from deepeval.test_run.api import LLMApiTestCase, ConversationalApiTestCase
from deepeval.tracing.api import ExpectationsData, MetricData


def expectations_metric_data():
    return MetricData(
        name="Expectations",
        score=1.0,
        success=True,
        evaluationCost=0.01,
        expectationsData=ExpectationsData(success=True, score=1.0),
    )


def make_run(kind):
    expectations = Expectations(must=["Answer"])
    run = tr_mod.TestRun()
    if kind == "single":
        case = LLMApiTestCase(
            name="case", input="Hi", order=0, expectations=expectations
        )
        case.update_metric_data(expectations_metric_data())
        run.test_cases = [case]
    else:
        case = ConversationalApiTestCase(
            name="conversation",
            success=True,
            metricsData=[],
            order=0,
            expectations=expectations,
        )
        case.update_metric_data(expectations_metric_data())
        run.conversational_test_cases = [case]
    return run


@pytest.mark.parametrize("kind", ["single", "conversation"])
def test_post_uploads_expectations_data(monkeypatch, kind):
    run = make_run(kind)
    api = Mock()
    api.send_request.return_value = ({"id": "run-id"}, "link")
    monkeypatch.setattr(tr_mod, "Api", Mock(return_value=api))
    monkeypatch.setattr(tr_mod, "get_is_running_deepeval", lambda: False)

    tr_mod.TestRunManager().post_test_run(run)

    body = api.send_request.call_args.kwargs["body"]
    [uploaded] = body.get("testCases", []) + body.get(
        "conversationalTestCases", []
    )
    assert "expectations" not in uploaded
    assert uploaded["metricsData"] == []
    assert uploaded["expectationsData"]["success"] is True


@pytest.mark.parametrize("kind", ["single", "conversation"])
def test_expectations_are_not_a_run_level_metric(kind):
    run = make_run(kind)
    valid_scores = run.construct_metrics_scores()

    assert run.metrics_scores == []
    # An expectations-only run still counts as having scored.
    assert valid_scores == 1


def test_wrap_up_with_expectations_uploads(monkeypatch):
    manager = tr_mod.TestRunManager()
    manager.disable_request = False
    run = make_run("single")
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
