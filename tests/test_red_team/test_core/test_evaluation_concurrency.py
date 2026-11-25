import asyncio

import pytest
from deepeval.models import DeepEvalBaseLLM

from deepeval.red_team.red_teamer import RedTeamer
from deepeval.red_team.test_case import RTTestCase
from deepeval.red_team.vulnerabilities import Bias


class OfflineModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def get_model_name(self):
        return "offline-concurrency-test"

    def generate(self, *args, **kwargs):
        raise AssertionError("No provider generation should run")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("No provider generation should run")


@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrent", [1, 2, 4, 16])
@pytest.mark.parametrize("failure", [None, "target", "judge"])
async def test_async_evaluation_bounds_individual_attacks(
    monkeypatch, max_concurrent, failure
):
    vulnerability = Bias(types=["gender", "race"])
    cases = [
        RTTestCase(
            vulnerability=vulnerability.get_name(),
            vulnerability_type=kind,
            input=f"{kind.value}-{index}",
            attack_method="Base64",
        )
        for kind in vulnerability.get_types()
        for index in range(6)
    ]
    model = OfflineModel()
    red_teamer = RedTeamer(
        simulator_model=model,
        evaluation_model=model,
        max_concurrent=max_concurrent,
    )

    async def simulate(**kwargs):
        return cases

    monkeypatch.setattr(red_teamer.attack_simulator, "a_simulate", simulate)
    active = 0
    peak = 0
    visited = []

    async def target(input):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        visited.append(input)
        try:
            await asyncio.sleep(0)
            if failure == "target" and input.endswith("-0"):
                raise RuntimeError("target unavailable")
            return "offline response"
        finally:
            active -= 1

    class OfflineMetric:
        score = 1.0
        reason = "offline verdict"
        evaluation_cost = None

        async def a_measure(self, case):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                if failure == "judge" and case.input.endswith("-0"):
                    raise RuntimeError("judge unavailable")
            finally:
                active -= 1

    monkeypatch.setattr(Bias, "_get_metric", lambda *args: OfflineMetric())
    result = await asyncio.wait_for(
        red_teamer.a_red_team(
            model_callback=target,
            vulnerabilities=[vulnerability],
            ignore_errors=True,
            _print_assessment=False,
            _upload_to_confident=False,
        ),
        timeout=10,
    )
    assert len(result.test_cases) == len(cases)
    assert sorted(visited) == sorted(case.input for case in cases)
    assert active == 0
    assert peak == min(max_concurrent, len(cases))
    errors = [case for case in result.test_cases if case.error]
    assert len(errors) == (0 if failure is None else 2)
    assert all(case.score == 1.0 for case in cases if not case.error)
