from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deepeval.dataset.golden import Golden
from deepeval.optimizer.algorithms import COPRO
from deepeval.optimizer.types import (
    IterationLogEntry,
    OptimizationReport,
    PromptConfigSnapshot,
    PromptConfiguration,
)
from deepeval.prompt.prompt import Prompt


def _goldens(n: int = 3) -> list[Golden]:
    return [Golden(input=f"q{i}", expected_output=f"a{i}") for i in range(n)]


def test_copro_sample_minibatch_respects_size() -> None:
    runner = COPRO(depth=1, breadth=1, minibatch_size=2, random_state=0)
    g = _goldens(5)
    mb = runner._sample_minibatch(g)
    assert len(mb) == 2
    assert all(x in g for x in mb)


def test_copro_sample_minibatch_returns_all_when_small() -> None:
    runner = COPRO(minibatch_size=10, random_state=0)
    g = _goldens(2)
    assert runner._sample_minibatch(g) == g


def test_copro_extract_optimized_set_picks_highest_mean() -> None:
    runner = COPRO(random_state=0)
    low = PromptConfiguration.new(
        prompts={COPRO.SINGLE_MODULE_ID: Prompt(text_template="low")}
    )
    high = PromptConfiguration.new(
        prompts={COPRO.SINGLE_MODULE_ID: Prompt(text_template="high")}
    )
    runner.pareto_score_table[low.id] = [0.2, 0.2]
    runner.pareto_score_table[high.id] = [0.9, 0.7]
    assert runner._extract_optimized_set() == high.id


def test_copro_execute_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    goldens = _goldens(3)
    runner = COPRO(depth=1, breadth=1, minibatch_size=2, random_state=0)
    runner.optimizer_model = MagicMock()

    proposer = MagicMock()
    proposer.propose_bootstrap.return_value = [
        Prompt(text_template="candidate CHILD"),
    ]
    proposer.propose_from_history.return_value = []

    def _fake_init(self: COPRO) -> None:
        self.proposer = proposer

    monkeypatch.setattr(COPRO, "_init_components", _fake_init)

    scorer = MagicMock()
    scorer.score_pareto.return_value = [1.0, 1.0]
    runner.scorer = scorer

    def _fake_eval(self: COPRO, config, minibatch) -> tuple[float, str]:
        return (0.95, "feedback")

    monkeypatch.setattr(COPRO, "_evaluate_candidate", _fake_eval)

    best, report = runner.execute(Prompt(text_template="root"), goldens)

    assert isinstance(best, Prompt)
    assert isinstance(report, OptimizationReport)
    assert report.optimization_id
    assert report.best_id in runner.prompt_configurations_by_id
    scorer.score_pareto.assert_called()
    proposer.propose_bootstrap.assert_called_once()


@pytest.mark.asyncio
async def test_copro_a_execute_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    goldens = _goldens(3)
    runner = COPRO(depth=1, breadth=1, minibatch_size=2, random_state=0)
    runner.optimizer_model = MagicMock()

    proposer = MagicMock()
    proposer.a_propose_bootstrap = AsyncMock(
        return_value=[Prompt(text_template="candidate CHILD")]
    )
    proposer.a_propose_from_history = AsyncMock(return_value=[])

    def _fake_init(self: COPRO) -> None:
        self.proposer = proposer

    monkeypatch.setattr(COPRO, "_init_components", _fake_init)

    scorer = MagicMock()
    scorer.a_score_pareto = AsyncMock(return_value=[1.0, 1.0])
    runner.scorer = scorer

    async def _fake_a_eval(self, config, minibatch):
        return (0.95, "feedback")

    monkeypatch.setattr(COPRO, "_a_evaluate_candidate", _fake_a_eval)

    best, report = await runner.a_execute(Prompt(text_template="root"), goldens)

    assert isinstance(best, Prompt)
    assert isinstance(report, OptimizationReport)
    assert report.optimization_id
    assert report.best_id in runner.prompt_configurations_by_id
    scorer.a_score_pareto.assert_awaited()
    proposer.a_propose_bootstrap.assert_awaited_once()


def test_copro_generate_summary_table_renders_iteration_log() -> None:
    runner = COPRO(random_state=0)
    runner._iteration_log = [
        IterationLogEntry(
            iteration=1,
            outcome="evaluated",
            before=0.0,
            after=0.5,
            reason="note",
            elapsed=0.1,
        )
    ]
    snap = PromptConfigSnapshot(
        parent=None,
        prompts={COPRO.SINGLE_MODULE_ID: Prompt(text_template="x")},
    )
    report = OptimizationReport(
        optimization_id="opt-1",
        best_id="abc",
        accepted_iterations=[],
        pareto_scores={"abc": [0.5, 0.6]},
        parents={"abc": None},
        prompt_configurations={"abc": snap},
    )
    tables = runner.generate_summary_table(report)
    assert len(tables) >= 1
