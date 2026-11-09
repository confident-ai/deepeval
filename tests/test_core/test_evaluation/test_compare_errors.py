import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import ArenaGEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)


compare_module = importlib.import_module("deepeval.evaluate.compare")


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def invoke(request):
    async def run(metric, **options):
        if request.param:
            return await compare_module._a_handle_metric_measurement(
                metric, None, **options
            )
        return compare_module._handle_metric_measurement(
            metric, None, **options
        )

    return run


def failing_metric(error):
    return SimpleNamespace(
        measure=Mock(side_effect=error),
        a_measure=AsyncMock(side_effect=error),
        error=None,
        success=None,
    )


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_arena_error_handling(invoke, error_type, ignore_errors):
    error = error_type("judge failed")
    metric = failing_metric(error)

    if ignore_errors:
        winner = await invoke(
            metric, ignore_errors=True, skip_on_missing_params=False
        )
        assert winner is None
        assert metric.error == "judge failed"
        assert metric.success is False
    else:
        with pytest.raises(error_type) as caught:
            await invoke(
                metric, ignore_errors=False, skip_on_missing_params=False
            )
        assert caught.value is error

    assert metric.measure.call_count + metric.a_measure.await_count == 1


@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_arena_cancellation_propagates(invoke, ignore_errors):
    error = asyncio.CancelledError()
    metric = failing_metric(error)

    with pytest.raises(asyncio.CancelledError) as caught:
        await invoke(
            metric, ignore_errors=ignore_errors, skip_on_missing_params=False
        )

    assert caught.value is error
    assert metric.error is None


@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_arena_missing_parameters_still_skip(invoke, ignore_errors):
    metric = failing_metric(MissingTestCaseParamsError("missing input"))

    winner = await invoke(
        metric, ignore_errors=ignore_errors, skip_on_missing_params=True
    )

    assert winner is None
    assert metric.error is None
    assert metric.success is None


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_compare_continues_after_ignored_error(monkeypatch, run_async):
    judge = Mock(spec=DeepEvalBaseLLM)
    judge.get_model_name.return_value = "stub-judge"
    metric = ArenaGEval(
        name="Correctness",
        criteria="Choose the correct answer",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
    )
    cases = [
        ArenaTestCase(
            contestants=[
                Contestant(
                    name=name,
                    test_case=LLMTestCase(input=label, actual_output=name),
                )
                for name in ["A", "B"]
            ]
        )
        for label in ["bad", "good"]
    ]
    calls = []

    def measure(self, test_case, **kwargs):
        label = test_case.contestants[0].test_case.input
        calls.append(label)
        if label == "bad":
            raise ValueError("judge failed")
        self.winner = "A"
        self.success = True
        return self.winner

    async def a_measure(self, test_case, **kwargs):
        return measure(self, test_case, **kwargs)

    monkeypatch.setattr(ArenaGEval, "measure", measure)
    monkeypatch.setattr(ArenaGEval, "a_measure", a_measure)
    wrap_up = Mock()
    monkeypatch.setattr(compare_module, "wrap_up_experiment", wrap_up)

    winners = compare_module.compare(
        cases,
        metric,
        async_config=AsyncConfig(run_async=run_async, max_concurrent=1),
        display_config=DisplayConfig(show_indicator=True),
        error_config=ErrorConfig(ignore_errors=True),
    )

    assert winners == {"A": 1}
    assert calls == ["bad", "good"]
    for test_run in wrap_up.call_args.kwargs["test_runs"]:
        assert len(test_run.test_cases) == 2
        failed_row = test_run.test_cases[0].metrics_data[0]
        assert failed_row.error == "judge failed"
        assert failed_row.success is False
        assert failed_row.score is None
        assert test_run.test_cases[1].metrics_data[0].error is None
