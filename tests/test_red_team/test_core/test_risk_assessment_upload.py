"""A risk assessment run locally reaches Confident AI with the same detail as
one run on the platform: the probes a multi-turn attack sent (rendered in the
test case's Algorithm tab), its simulation and evaluation costs, and the model
that simulated the attacks. No test here makes a model call.
"""

import pytest

import deepeval.red_team.red_teamer.red_teamer as red_teamer_module
from deepeval.models import DeepEvalBaseLLM
from deepeval.red_team.attacks.attack_simulator import AttackSimulator
from deepeval.red_team.attacks.multi_turn import LinearJailbreaking
from deepeval.red_team.attacks.multi_turn.progression import (
    ProgressionResult,
    StopReason,
)
from deepeval.red_team.red_teamer import RedTeamer
from deepeval.red_team.red_teamer.api import map_test_case_to_api
from deepeval.red_team.red_teamer.risk_assessment import (
    RiskAssessment,
    construct_risk_assessment_overview,
)
from deepeval.red_team.test_case import RTTestCase, RTTurn
from deepeval.red_team.test_case.test_case import Probe
from deepeval.red_team.vulnerabilities.bias.types import BiasType


class _StubSimulator(DeepEvalBaseLLM):
    """Stands in for the simulator model. Fails the test if it is called."""

    def __init__(self):
        super().__init__("stub-simulator")

    def load_model(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs):
        raise AssertionError("no model call expected")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("no model call expected")

    def get_model_name(self, *args, **kwargs):
        return "stub-simulator"


def _probes():
    return [
        Probe(
            id=0,
            input="opening attack",
            output=RTTurn(role="assistant", content="refused"),
            committed=False,
            depth=0,
            latency=0.4,
            simulation_cost=0.001,
            score=1.0,
            reason="held firm",
        ),
        Probe(
            id=1,
            parent_id=0,
            input="reframed attack",
            output=RTTurn(role="assistant", content="complied"),
            turn_level_attack="Roleplay",
            committed=True,
            depth=1,
            latency=0.6,
            simulation_cost=0.002,
            score=0.0,
            reason="complied",
        ),
    ]


def _progression_result(probes, error=None):
    return ProgressionResult(
        turns=[
            RTTurn(role="user", content="reframed attack"),
            RTTurn(role="assistant", content="complied"),
        ],
        stop_reason=(
            StopReason.SIMULATION_ERROR if error else StopReason.SHIFT_DETECTED
        ),
        probes=probes,
        error=error,
    )


def _multi_turn_test_case():
    return RTTestCase(
        vulnerability="Bias",
        vulnerability_type=BiasType.GENDER,
        risk_category="Responsible AI",
        input="opening attack",
    )


def _simulator():
    return AttackSimulator(
        purpose="customer support",
        max_concurrent=1,
        simulator_model=_StubSimulator(),
    )


def test_simulator_keeps_the_probes_of_a_multi_turn_attack(monkeypatch):
    probes = _probes()
    attack = LinearJailbreaking()
    monkeypatch.setattr(attack, "run", lambda **_: _progression_result(probes))

    test_case = _simulator().enhance_attack(
        attack, _multi_turn_test_case(), ignore_errors=False
    )

    assert test_case.probes == probes


@pytest.mark.asyncio
async def test_async_simulator_keeps_the_probes_of_a_multi_turn_attack(
    monkeypatch,
):
    probes = _probes()
    attack = LinearJailbreaking()

    async def a_run(**_):
        return _progression_result(probes)

    monkeypatch.setattr(attack, "a_run", a_run)

    test_case = await _simulator().a_enhance_attack(
        attack, _multi_turn_test_case(), ignore_errors=False
    )

    assert test_case.probes == probes


def test_an_errored_progression_still_keeps_its_probes(monkeypatch):
    probes = _probes()
    attack = LinearJailbreaking()
    monkeypatch.setattr(
        attack,
        "run",
        lambda **_: _progression_result(probes, error=RuntimeError("boom")),
    )

    test_case = _simulator().enhance_attack(
        attack, _multi_turn_test_case(), ignore_errors=True
    )

    assert test_case.error is not None
    assert test_case.probes == probes


def test_posted_risk_assessment_carries_probes_costs_and_simulation_model(
    monkeypatch,
):
    posted = {}

    class _CapturingApi:
        def send_request(self, method, endpoint, body=None, **_):
            posted["body"] = body
            return {"id": "risk-assessment"}, "https://app.confident-ai.com"

    monkeypatch.setattr(red_teamer_module, "is_confident", lambda: True)
    monkeypatch.setattr(red_teamer_module, "Api", _CapturingApi)
    monkeypatch.setattr(red_teamer_module.webbrowser, "open", lambda *_: None)

    test_case = _multi_turn_test_case()
    test_case.turns = _progression_result([]).turns
    test_case.probes = _probes()
    test_case.score = 0.0
    test_case.simulation_cost = 0.003
    test_case.evaluation_cost = 0.0005

    red_teamer = RedTeamer(
        simulator_model=_StubSimulator(), evaluation_model=_StubSimulator()
    )
    red_teamer.risk_assessment = RiskAssessment(
        overview=construct_risk_assessment_overview(
            [test_case], run_duration=1.0
        ),
        test_cases=[test_case],
    )
    red_teamer._post_risk_assessment()

    sent = posted["body"]["testCases"][0]
    assert sent["simulationModel"] == "stub-simulator"
    assert sent["simulationCost"] == 0.003
    assert sent["evaluationCost"] == 0.0005
    assert sent["probes"] == [
        {
            "id": 0,
            "input": "opening attack",
            "output": "refused",
            "committed": False,
            "latency": 0.4,
            "simulationCost": 0.001,
            "score": 1.0,
            "reason": "held firm",
        },
        {
            "id": 1,
            "parentId": 0,
            "input": "reframed attack",
            "output": "complied",
            "turnLevelAttack": "Roleplay",
            "committed": True,
            "latency": 0.6,
            "simulationCost": 0.002,
            "score": 0.0,
            "reason": "complied",
        },
    ]


def test_a_single_turn_test_case_sends_no_probes():
    test_case = RTTestCase(
        vulnerability="Bias",
        vulnerability_type=BiasType.GENDER,
        risk_category="Responsible AI",
        input="attack",
        actual_output="reply",
        score=1.0,
    )

    payload = map_test_case_to_api(test_case, 0).model_dump(
        by_alias=True, exclude_none=True
    )

    assert "probes" not in payload
