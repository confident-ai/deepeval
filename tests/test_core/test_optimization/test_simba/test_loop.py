from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepeval.dataset.golden import ConversationalGolden, Golden
from deepeval.optimizer.algorithms import SIMBA
from deepeval.optimizer.types import (
    IterationLogEntry,
    OptimizationReport,
    PromptConfigSnapshot,
    SimbaTraceRecord,
)
from deepeval.prompt.prompt import Prompt
from deepeval.test_case import Turn


def _goldens(n: int = 3) -> list[Golden]:
    return [Golden(input=f"q{i}", expected_output=f"a{i}") for i in range(n)]


def test_simba_golden_expected_text() -> None:
    g = Golden(input="x", expected_output="eo")
    assert SIMBA._golden_expected_text(g) == "eo"
    assert SIMBA._golden_expected_text(Golden(input="x")) is None

    cg = ConversationalGolden(scenario="s", expected_outcome="out")
    assert SIMBA._golden_expected_text(cg) == "out"


def test_simba_extract_inputs_golden_and_conversational() -> None:
    runner = SIMBA(random_state=0)
    g = Golden(input="plain")
    assert runner._extract_inputs(g) == "plain"

    cg = ConversationalGolden(
        scenario="sc",
        turns=[
            Turn(role="user", content=" hi "),
            Turn(role="assistant", content="bot"),
        ],
    )
    assert runner._extract_inputs(cg) == " hi "


def test_simba_sample_minibatch_respects_size() -> None:
    runner = SIMBA(minibatch_size=2, random_state=0)
    g = _goldens(5)
    mb = runner._sample_minibatch(g)
    assert len(mb) == 2


def test_simba_generate_summary_table_renders_iteration_log() -> None:
    runner = SIMBA(random_state=0)
    runner._iteration_log = [
        IterationLogEntry(
            iteration=1,
            outcome="accepted",
            before=0.0,
            after=1.0,
            reason="ok",
            elapsed=0.05,
        )
    ]
    snap = PromptConfigSnapshot(
        parent=None,
        prompts={SIMBA.SINGLE_MODULE_ID: Prompt(text_template="x")},
    )
    report = OptimizationReport(
        optimization_id="opt-1",
        best_id="abc",
        accepted_iterations=[],
        pareto_scores={"abc": [1.0]},
        parents={"abc": None},
        prompt_configurations={"abc": snap},
    )
    tables = runner.generate_summary_table(report)
    assert len(tables) >= 1


def test_simba_execute_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    goldens = _goldens(3)
    runner = SIMBA(
        iterations=1,
        minibatch_size=1,
        num_candidates=1,
        num_samples=2,
        minibatch_full_eval_steps=1,
        random_state=0,
    )
    runner.optimizer_model = MagicMock()

    real_rng = random.Random(42)
    mock_rng = MagicMock()
    mock_rng.sample.side_effect = lambda g, k: real_rng.sample(g, k)
    mock_rng.choice.return_value = "rule"
    runner.random_state = mock_rng

    proposer = MagicMock()
    proposer.rewrite_from_introspection.return_value = Prompt(
        text_template="improved CHILD"
    )

    def _fake_init(self: SIMBA) -> None:
        self.proposer = proposer

    monkeypatch.setattr(SIMBA, "_init_components", _fake_init)

    scorer = MagicMock()
    scorer.score_minibatch.return_value = 0.99
    scorer.score_pareto.return_value = [1.0]
    runner.scorer = scorer

    trace_calls: list[int] = []

    def _fake_trace(self: SIMBA, cfg, golden) -> SimbaTraceRecord:
        trace_calls.append(1)
        score = 1.0 if len(trace_calls) % 2 == 1 else 0.1
        return SimbaTraceRecord(
            output=f"o{len(trace_calls)}", score=score, feedback="f"
        )

    monkeypatch.setattr(SIMBA, "_execute_trace", _fake_trace)

    best, report = runner.execute(Prompt(text_template="root"), goldens)

    assert isinstance(best, Prompt)
    assert isinstance(report, OptimizationReport)
    assert report.optimization_id
    assert "CHILD" in (best.text_template or "")
    proposer.rewrite_from_introspection.assert_called()
    scorer.score_pareto.assert_called()


@pytest.mark.asyncio
async def test_simba_a_execute_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    goldens = _goldens(3)
    runner = SIMBA(
        iterations=1,
        minibatch_size=1,
        num_candidates=1,
        num_samples=2,
        minibatch_full_eval_steps=1,
        random_state=0,
    )
    runner.optimizer_model = MagicMock()

    real_rng = random.Random(42)
    mock_rng = MagicMock()
    mock_rng.sample.side_effect = lambda g, k: real_rng.sample(g, k)
    mock_rng.choice.return_value = "rule"
    runner.random_state = mock_rng

    proposer = MagicMock()
    proposer.a_rewrite_from_introspection = AsyncMock(
        return_value=Prompt(text_template="async CHILD")
    )

    def _fake_init(self: SIMBA) -> None:
        self.proposer = proposer

    monkeypatch.setattr(SIMBA, "_init_components", _fake_init)

    scorer = MagicMock()
    scorer.a_score_minibatch = AsyncMock(return_value=0.99)
    scorer.a_score_pareto = AsyncMock(return_value=[1.0])
    runner.scorer = scorer

    trace_calls: list[int] = []

    async def _fake_a_trace(self: SIMBA, cfg, golden) -> SimbaTraceRecord:
        trace_calls.append(1)
        score = 1.0 if len(trace_calls) % 2 == 1 else 0.1
        return SimbaTraceRecord(
            output=f"a{len(trace_calls)}", score=score, feedback="f"
        )

    monkeypatch.setattr(SIMBA, "_a_execute_trace", _fake_a_trace)

    best, report = await runner.a_execute(Prompt(text_template="root"), goldens)

    assert isinstance(best, Prompt)
    assert isinstance(report, OptimizationReport)
    assert report.optimization_id
    assert "CHILD" in (best.text_template or "")
    proposer.a_rewrite_from_introspection.assert_awaited()
    scorer.a_score_pareto.assert_awaited()
