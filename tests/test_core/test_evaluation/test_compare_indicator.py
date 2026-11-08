from contextlib import nullcontext
from importlib import import_module

import pytest

from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.test_case import ArenaTestCase, Contestant, LLMTestCase


@pytest.mark.parametrize("show_indicator", [True, False])
def test_async_compare_runs_cases_with_or_without_indicator(
    monkeypatch, show_indicator
):
    compare_module = import_module("deepeval.evaluate.compare")
    calls = []

    class FakeArenaGEval:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "Helpfulness")
            self.evaluation_params = kwargs.get("evaluation_params", [])
            self.criteria = kwargs.get("criteria", "Choose the better answer")
            self.evaluation_steps = kwargs.get("evaluation_steps")
            self.model = kwargs.get("model")
            self.verbose_mode = kwargs.get("verbose_mode", False)

        async def a_measure(self, test_case, **kwargs):
            calls.append(test_case)
            return "Bot B"

    case = ArenaTestCase(
        contestants=[
            Contestant(
                name="Bot A",
                test_case=LLMTestCase(input="Question", actual_output="A"),
            ),
            Contestant(
                name="Bot B",
                test_case=LLMTestCase(input="Question", actual_output="B"),
            ),
        ]
    )
    monkeypatch.setattr(compare_module, "ArenaGEval", FakeArenaGEval)
    monkeypatch.setattr(
        compare_module, "capture_evaluation_run", lambda *args: nullcontext()
    )
    monkeypatch.setattr(
        compare_module, "update_test_run_map", lambda **kwargs: None
    )
    monkeypatch.setattr(
        compare_module, "process_test_runs", lambda **kwargs: None
    )
    monkeypatch.setattr(
        compare_module, "wrap_up_experiment", lambda **kwargs: None
    )

    result = compare_module.compare(
        test_cases=[case],
        metric=FakeArenaGEval(),
        async_config=AsyncConfig(run_async=True),
        display_config=DisplayConfig(show_indicator=show_indicator),
    )

    assert calls == [case]
    assert result == {"Bot B": 1}
