from __future__ import annotations

import pytest

from deepeval.errors import DeepEvalError
from deepeval.prompt.prompt import Prompt
from deepeval.prompt.api import PromptMessage
from deepeval.optimization.types import PromptConfiguration
from deepeval.optimization.copro.configs import COPROConfig
from deepeval.optimization.copro.loop import COPRORunner

from tests.test_core.stubs import (
    StubScoringAdapter,
    SuffixRewriter,
    DummyProgress,
)


def _make_runner(
    *,
    iterations: int = 3,
    population_size: int = 4,
    proposals_per_step: int = 2,
    full_eval_every: int | None = 2,
):
    """
    Helper to construct a COPRORunner wired with the shared stubs:

    - StubScoringAdapter: scores prompts with 'CHILD' higher than root.
    - SuffixRewriter: appends ' CHILD' to the prompt text.
    """
    config = COPROConfig(
        iterations=iterations,
        population_size=population_size,
        proposals_per_step=proposals_per_step,
        full_eval_every=full_eval_every,
        random_seed=0,
        minibatch_size=2,  # keep minibatch behaviour deterministic in tests
    )
    scoring = StubScoringAdapter()
    runner = COPRORunner(config=config, scoring_adapter=scoring)
    runner._rewriter = SuffixRewriter(suffix=" CHILD")
    return runner, scoring


def test_execute_requires_at_least_one_golden():
    """COPRORunner.execute must reject an empty golden list."""
    runner, _ = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        runner.execute(prompt=prompt, goldens=[])


@pytest.mark.asyncio
async def test_a_execute_requires_at_least_one_golden():
    """COPRORunner.a_execute must reject an empty golden list."""
    runner, _ = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        await runner.a_execute(prompt=prompt, goldens=[])


def test_execute_requires_scoring_adapter():
    """
    If no scoring_adapter is attached, COPRORunner.execute must fail
    via _ensure_scoring_adapter.
    """
    config = COPROConfig()
    runner = COPRORunner(config=config, scoring_adapter=None)
    prompt = Prompt(text_template="ROOT")
    goldens = ["g1"]

    with pytest.raises(DeepEvalError):
        runner.execute(prompt=prompt, goldens=goldens)


def test_add_prompt_configuration_prunes_worst_candidate():
    """
    _add_prompt_configuration must enforce population_size by pruning
    the worst-scoring candidate while always keeping the best.
    """
    config = COPROConfig(
        population_size=2,
        iterations=1,
        random_seed=0,
    )
    runner = COPRORunner(config=config, scoring_adapter=None)

    # Three candidates with different mean scores
    c1 = PromptConfiguration.new(
        prompts={COPRORunner.SINGLE_MODULE_ID: Prompt(text_template="C1")}
    )
    c2 = PromptConfiguration.new(
        prompts={COPRORunner.SINGLE_MODULE_ID: Prompt(text_template="C2")}
    )
    c3 = PromptConfiguration.new(
        prompts={COPRORunner.SINGLE_MODULE_ID: Prompt(text_template="C3")}
    )

    # Seed surrogate stats BEFORE adding, so pruning logic sees real scores
    runner._minibatch_score_sums[c1.id] = 0.1
    runner._minibatch_score_counts[c1.id] = 1

    runner._minibatch_score_sums[c2.id] = 0.5
    runner._minibatch_score_counts[c2.id] = 1

    runner._minibatch_score_sums[c3.id] = 0.9
    runner._minibatch_score_counts[c3.id] = 1

    runner._add_prompt_configuration(c1)
    runner._add_prompt_configuration(c2)
    # This call should trigger pruning (population_size=2)
    runner._add_prompt_configuration(c3)

    # The best candidate (c3) must be kept, and the worst (c1) pruned.
    ids = set(runner.prompt_configurations_by_id.keys())
    assert len(ids) == 2
    assert c3.id in ids
    assert c2.id in ids
    assert c1.id not in ids


def test_execute_accepts_children_and_respects_population():
    """
    A full COPRO run should:
    - Use minibatch scoring and feedback from StubScoringAdapter.
    - Accept children whose text contains 'CHILD' (higher score).
    - Respect the population_size bound via _add_prompt_configuration.
    """
    runner, scoring = _make_runner(
        iterations=3,
        population_size=2,
        proposals_per_step=3,
        full_eval_every=1,
    )
    prompt = Prompt(text_template="ROOT")
    goldens = ["g1", "g2", "g3", "g4"]

    best_prompt, report = runner.execute(prompt=prompt, goldens=goldens)

    # Best prompt should be an improved child (SuffixRewriter appends ' CHILD')
    assert isinstance(best_prompt, Prompt)
    assert "CHILD" in (best_prompt.text_template or "")

    # Optimization report basic shape
    assert isinstance(report, dict)
    assert "best_id" in report
    assert "optimization_id" in report
    assert "accepted_iterations" in report

    # Population bound respected
    assert (
        len(runner.prompt_configurations_by_id) <= runner.config.population_size
    )

    # Surrogate stats should have been populated
    assert runner._minibatch_score_counts
    # StubScoringAdapter must have been used
    assert scoring.score_calls  # minibatch_score used at least once
    assert scoring.feedback_calls  # minibatch_feedback used at least once
    assert scoring.pareto_calls  # full_eval_every=1 triggers score_on_pareto


@pytest.mark.asyncio
async def test_a_execute_uses_async_paths_and_accepts_children():
    """
    Async COPRO run should:
    - Use a_minibatch_* and a_score_on_pareto paths on StubScoringAdapter.
    - Accept children and produce an improved best prompt.
    """
    runner, scoring = _make_runner(
        iterations=2,
        population_size=3,
        proposals_per_step=2,
        full_eval_every=1,
    )
    prompt = Prompt(text_template="ROOT")
    goldens = ["g1", "g2", "g3"]

    best_prompt, report = await runner.a_execute(prompt=prompt, goldens=goldens)

    assert isinstance(best_prompt, Prompt)
    assert "CHILD" in (best_prompt.text_template or "")

    assert isinstance(report, dict)
    assert "best_id" in report

    # Async scoring methods should have been exercised
    assert scoring.a_score_calls  # a_minibatch_score
    assert scoring.a_feedback_calls  # a_minibatch_feedback
    assert scoring.a_pareto_calls  # a_score_on_pareto


def test_prompts_equivalent_detects_text_and_list_prompts():
    """
    _prompts_equivalent should treat prompts with identical trimmed text
    (or identical LIST messages) as equivalent, and different ones as not.
    """
    cfg = COPROConfig()
    runner = COPRORunner(config=cfg, scoring_adapter=None)

    # TEXT prompts
    p1 = Prompt(text_template="  hello world  ")
    p2 = Prompt(text_template="hello world")
    p3 = Prompt(text_template="hello WORLD!!!")

    assert runner._prompts_equivalent(p1, p2) is True
    assert runner._prompts_equivalent(p1, p3) is False

    # LIST prompts: same roles + trimmed content => equivalent

    msgs1 = [
        PromptMessage(role="user", content="  hi  "),
        PromptMessage(role="assistant", content="there"),
    ]
    msgs2 = [
        PromptMessage(role="user", content="hi"),
        PromptMessage(role="assistant", content="there"),
    ]
    msgs3 = [
        PromptMessage(role="user", content="hello"),
        PromptMessage(role="assistant", content="there"),
    ]

    # NOTE: Prompt infers LIST type from messages_template; no `type=` kwarg.
    lp1 = Prompt(messages_template=msgs1)
    lp2 = Prompt(messages_template=msgs2)
    lp3 = Prompt(messages_template=msgs3)

    assert runner._prompts_equivalent(lp1, lp2) is True
    assert runner._prompts_equivalent(lp1, lp3) is False


def test_update_progress_and_error_use_status_callback():
    """
    _update_progress and _update_error should forward structured events to
    the status_callback, allowing PromptOptimizer to drive a progress bar.
    """
    cfg = COPROConfig(iterations=2)
    runner = COPRORunner(config=cfg, scoring_adapter=None)

    progress = DummyProgress()

    def status_callback(kind, step_index, total_steps, detail):
        # Mimic PromptOptimizer._on_status behaviour by recording updates
        progress.update(
            "task",
            kind=kind,
            step_index=step_index,
            total_steps=total_steps,
            detail=detail,
        )

    runner.status_callback = status_callback

    runner._update_progress(
        total_iterations=2, iteration=1, remaining_iterations=1, elapsed=0.123
    )
    runner._update_error(
        total_iterations=2, iteration=1, exc=RuntimeError("boom")
    )

    # We don't assert exact payload, just that our DummyProgress saw both calls.
    assert len(progress.records) == 2
