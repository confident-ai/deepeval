import importlib

import pytest

from deepeval.red_team.orchestration import AgenticRedTeamer
from deepeval.red_team.test_case.test_case import RTTurn

module = importlib.import_module(
    "deepeval.red_team.orchestration.agentic_red_teamer"
)


class StubAgent:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.criteria = kwargs["criteria"]
        self.max_turns = kwargs.get("max_turns", 5)
        self.met_criteria = []
        self.verdicts = []
        self.rounds = []
        self.recon_result = None
        type(self).last = self

    async def a_run(self):
        return f"result-from-{type(self).__name__}"

    def run(self):
        return f"sync-result-from-{type(self).__name__}"


class StubSequential(StubAgent):
    pass


class StubOrchestration(StubAgent):
    pass


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(module, "SequentialAgent", StubSequential)
    monkeypatch.setattr(module, "OrchestrationAgent", StubOrchestration)
    StubSequential.last = None
    StubOrchestration.last = None


def turns_callback(message, turns=None):
    return RTTurn(role="assistant", content="ok")


def no_turns_callback(message):
    return RTTurn(role="assistant", content="ok")


def test_sequential_mode_uses_the_sequential_agent(stubs):
    teamer = AgenticRedTeamer(
        "t", "s", ["c1", "c2"], mode="sequential", show_progress=False
    )
    result = teamer.run(turns_callback)
    assert result == "result-from-StubSequential"
    assert StubSequential.last.kwargs["criteria"] == ["c1", "c2"]
    assert StubOrchestration.last is None


def test_parallel_mode_uses_the_orchestration_agent(stubs):
    teamer = AgenticRedTeamer(
        "t", "s", ["c1"], mode="parallel", show_progress=False
    )
    result = teamer.run(turns_callback)
    assert result == "result-from-StubOrchestration"
    assert StubOrchestration.last is not None


def test_parallel_is_the_default_mode(stubs):
    teamer = AgenticRedTeamer("t", "s", ["c1", "c2"], show_progress=False)
    result = teamer.run(turns_callback)
    assert result == "result-from-StubOrchestration"
    assert StubOrchestration.last.kwargs["criteria"] == ["c1", "c2"]
    assert StubSequential.last is None


def test_parallel_requires_a_turns_accepting_callback(stubs):
    teamer = AgenticRedTeamer(
        "t", "s", ["c1"], mode="parallel", show_progress=False
    )
    with pytest.raises(ValueError, match="turns"):
        teamer.run(no_turns_callback)


def test_sequential_accepts_a_single_arg_callback(stubs):
    teamer = AgenticRedTeamer(
        "t", "s", ["c1"], mode="sequential", show_progress=False
    )
    result = teamer.run(no_turns_callback)
    assert result == "result-from-StubSequential"


def test_invalid_mode_raises_at_construction(stubs):
    with pytest.raises(ValueError, match="mode"):
        AgenticRedTeamer("t", "s", ["c1"], mode="nope", show_progress=False)


def test_max_turns_is_passed_through(stubs):
    teamer = AgenticRedTeamer(
        "t", "s", ["c1"], max_turns=7, show_progress=False
    )
    teamer.run(turns_callback)
    assert StubOrchestration.last.kwargs["max_turns"] == 7


def test_max_turns_omitted_uses_agent_default(stubs):
    AgenticRedTeamer("t", "s", ["c1"], show_progress=False).run(turns_callback)
    assert "max_turns" not in StubOrchestration.last.kwargs


def test_progress_bar_runs_without_error(stubs):
    teamer = AgenticRedTeamer("t", "s", ["c1"], show_progress=True)
    result = teamer.run(turns_callback)
    assert result == "result-from-StubOrchestration"


def test_async_mode_false_runs_synchronously(stubs):
    teamer = AgenticRedTeamer(
        "t", "s", ["c1"], async_mode=False, show_progress=False
    )
    result = teamer.run(turns_callback)
    assert result == "sync-result-from-StubOrchestration"


def test_async_mode_true_is_the_default(stubs):
    teamer = AgenticRedTeamer("t", "s", ["c1"], show_progress=False)
    result = teamer.run(turns_callback)
    assert result == "result-from-StubOrchestration"
