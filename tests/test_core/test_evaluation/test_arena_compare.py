from contextlib import nullcontext
import asyncio
import importlib
import time

import pytest

from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import ArenaGEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.test_run.test_run import MetricScores, TestRun as DeepEvalTestRun


class DummyLLM(DeepEvalBaseLLM):
    def load_model(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs) -> str:
        return '{"winner": "Version 2", "reason": "deterministic"}'

    async def a_generate(self, *args, **kwargs) -> str:
        return '{"winner": "Version 2", "reason": "deterministic"}'

    def get_model_name(self, *args, **kwargs) -> str:
        return "dummy"


def build_arena_test_case(case_id: str) -> ArenaTestCase:
    return ArenaTestCase(
        contestants=[
            Contestant(
                name="Version 1",
                test_case=LLMTestCase(
                    input=case_id,
                    actual_output="Hey! how are you?",
                ),
            ),
            Contestant(
                name="Version 2",
                test_case=LLMTestCase(
                    input=case_id,
                    actual_output="Hello.",
                ),
            ),
        ]
    )


def build_arena_metric() -> ArenaGEval:
    return ArenaGEval(
        name="Friendly",
        criteria="Choose the more accurate contestant.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        model=DummyLLM(),
    )


def build_test_run_map(metric_name: str):
    test_run_map = {}
    for contestant_name in ("Version 1", "Version 2"):
        test_run = DeepEvalTestRun(
            identifier=contestant_name,
            test_passed=0,
            test_failed=0,
        )
        test_run.metrics_scores = [
            MetricScores(
                metric=metric_name,
                scores=[],
                passes=0,
                fails=0,
                errors=0,
            )
        ]
        test_run_map[contestant_name] = test_run
    return test_run_map


def test_compare_async_without_indicator_executes_arena_tests(monkeypatch):
    compare_module = importlib.import_module("deepeval.evaluate.compare")
    measure_calls = []
    wrap_up_payload = {}

    async def fake_a_measure(
        self,
        test_case,
        _show_indicator=True,
        _progress=None,
        _pbar_id=None,
    ):
        measure_calls.append(test_case)
        self.winner = "Version 2"
        self.reason = "deterministic winner"
        self.success = True
        self.evaluation_cost = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.verbose_logs = None
        return self.winner

    def fake_wrap_up_experiment(**kwargs):
        wrap_up_payload.update(kwargs)

    monkeypatch.setattr(
        compare_module,
        "capture_evaluation_run",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        compare_module.ArenaGEval,
        "a_measure",
        fake_a_measure,
    )
    monkeypatch.setattr(
        compare_module,
        "wrap_up_experiment",
        fake_wrap_up_experiment,
    )

    test_case = build_arena_test_case("Say hello.")
    metric = build_arena_metric()

    result = compare_module.compare(
        test_cases=[test_case],
        metric=metric,
        async_config=AsyncConfig(run_async=True, throttle_value=0),
        display_config=DisplayConfig(show_indicator=False),
    )

    assert result == {"Version 2": 1}
    assert measure_calls == [test_case]
    assert dict(wrap_up_payload["winner_counts"]) == {"Version 2": 1}
    assert {run.identifier for run in wrap_up_payload["test_runs"]} == {
        "Version 1",
        "Version 2",
    }
    assert all(len(run.test_cases) == 1 for run in wrap_up_payload["test_runs"])


@pytest.mark.asyncio
async def test_compare_async_without_indicator_cancels_pending_tasks_on_error(
    monkeypatch,
):
    compare_module = importlib.import_module("deepeval.evaluate.compare")
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()
    slow_completed = False

    async def fake_a_measure(
        self,
        test_case,
        _show_indicator=True,
        _progress=None,
        _pbar_id=None,
    ):
        nonlocal slow_completed
        case_input = test_case.contestants[0].test_case.input
        if case_input == "fail":
            await slow_started.wait()
            raise RuntimeError("arena failure")

        slow_started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            slow_cancelled.set()
            raise
        slow_completed = True
        return "Version 2"

    monkeypatch.setattr(
        compare_module.ArenaGEval,
        "a_measure",
        fake_a_measure,
    )

    metric = build_arena_metric()

    with pytest.raises(RuntimeError, match="arena failure"):
        await compare_module.a_execute_arena_test_cases(
            test_cases=[
                build_arena_test_case("fail"),
                build_arena_test_case("slow"),
            ],
            metric=metric,
            ignore_errors=False,
            verbose_mode=False,
            show_indicator=False,
            throttle_value=0,
            skip_on_missing_params=False,
            max_concurrent=2,
            test_run_map=build_test_run_map(metric.name),
        )

    assert slow_cancelled.is_set()
    assert slow_completed is False


@pytest.mark.asyncio
async def test_compare_async_without_indicator_stops_scheduling_after_error(
    monkeypatch,
):
    compare_module = importlib.import_module("deepeval.evaluate.compare")
    started_cases = []

    async def fake_a_measure(
        self,
        test_case,
        _show_indicator=True,
        _progress=None,
        _pbar_id=None,
    ):
        case_input = test_case.contestants[0].test_case.input
        started_cases.append(case_input)
        if case_input == "fail":
            raise RuntimeError("arena failure")

        await asyncio.sleep(3600)
        return "Version 2"

    monkeypatch.setattr(
        compare_module.ArenaGEval,
        "a_measure",
        fake_a_measure,
    )

    metric = build_arena_metric()

    start_time = time.perf_counter()
    with pytest.raises(RuntimeError, match="arena failure"):
        await compare_module.a_execute_arena_test_cases(
            test_cases=[
                build_arena_test_case("fail"),
                build_arena_test_case("slow-1"),
                build_arena_test_case("slow-2"),
                build_arena_test_case("slow-3"),
            ],
            metric=metric,
            ignore_errors=False,
            verbose_mode=False,
            show_indicator=False,
            throttle_value=1,
            skip_on_missing_params=False,
            max_concurrent=4,
            test_run_map=build_test_run_map(metric.name),
        )
    elapsed = time.perf_counter() - start_time

    assert started_cases == ["fail"]
    assert elapsed < 0.9
