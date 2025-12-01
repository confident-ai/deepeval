from __future__ import annotations

import pytest

from deepeval.errors import DeepEvalError
from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.prompt.prompt import Prompt
from deepeval.prompt.api import PromptMessage
from deepeval.optimization.types import PromptConfiguration
from deepeval.optimization.simba.configs import SIMBAConfig
from deepeval.optimization.simba.types import SIMBAStrategy
from deepeval.optimization.simba.loop import SIMBARunner

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
    max_demos_per_proposal: int = 3,
    demo_input_max_chars: int = 256,
):
    """
    Helper to construct a SIMBARunner wired with the shared stubs:

    - StubScoringAdapter: scores prompts with 'CHILD' higher than root.
    - SuffixRewriter: appends ' CHILD' to the prompt text.
    """
    config = SIMBAConfig(
        iterations=iterations,
        population_size=population_size,
        proposals_per_step=proposals_per_step,
        full_eval_every=full_eval_every,
        random_seed=0,
        minibatch_size=2,  # keep minibatch behaviour deterministic in tests
        max_demos_per_proposal=max_demos_per_proposal,
        demo_input_max_chars=demo_input_max_chars,
    )
    scoring = StubScoringAdapter()
    runner = SIMBARunner(config=config, scoring_adapter=scoring)
    runner._rewriter = SuffixRewriter(suffix=" CHILD")
    return runner, scoring


#########################
# Basic guardrail tests #
#########################


def test_execute_requires_at_least_one_golden():
    """SIMBARunner.execute must reject an empty golden list."""
    runner, _ = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        runner.execute(prompt=prompt, goldens=[])


@pytest.mark.asyncio
async def test_a_execute_requires_at_least_one_golden():
    """SIMBARunner.a_execute must reject an empty golden list."""
    runner, _ = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        await runner.a_execute(prompt=prompt, goldens=[])


def test_execute_requires_scoring_adapter():
    """
    If no scoring_adapter is attached, SIMBARunner.execute must fail
    via _ensure_scoring_adapter.
    """
    config = SIMBAConfig()
    runner = SIMBARunner(config=config, scoring_adapter=None)
    prompt = Prompt(text_template="ROOT")
    goldens = ["g1"]

    with pytest.raises(DeepEvalError):
        runner.execute(prompt=prompt, goldens=goldens)


#########################################
# Population management & pruning logic #
#########################################


def test_add_prompt_configuration_prunes_worst_candidate():
    """
    _add_prompt_configuration must enforce population_size by pruning
    the worst-scoring candidate while always keeping the best.
    """
    config = SIMBAConfig(
        population_size=2,
        iterations=1,
        random_seed=0,
    )
    runner = SIMBARunner(config=config, scoring_adapter=None)

    # Three candidates with different mean scores
    c1 = PromptConfiguration.new(
        prompts={SIMBARunner.SINGLE_MODULE_ID: Prompt(text_template="C1")}
    )
    c2 = PromptConfiguration.new(
        prompts={SIMBARunner.SINGLE_MODULE_ID: Prompt(text_template="C2")}
    )
    c3 = PromptConfiguration.new(
        prompts={SIMBARunner.SINGLE_MODULE_ID: Prompt(text_template="C3")}
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


######################################
# Main loop: sync & async behaviour  #
######################################


def test_execute_accepts_children_and_respects_population():
    """
    A full SIMBA run should:
    - Use minibatch scoring and feedback from StubScoringAdapter.
    - Accept children whose text contains 'CHILD' (higher score).
    - Respect the population_size bound via _add_prompt_configuration.
    """
    runner, scoring = _make_runner(
        iterations=3,
        population_size=2,
        proposals_per_step=3,
        full_eval_every=1,
        max_demos_per_proposal=2,
    )
    prompt = Prompt(text_template="ROOT")
    # For this test we don't need real Golden objects; the stub only cares
    # that `goldens` is indexable and non-empty.
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
    Async SIMBA run should:
    - Use a_minibatch_* and a_score_on_pareto paths on StubScoringAdapter.
    - Accept children and produce an improved best prompt.
    """
    runner, scoring = _make_runner(
        iterations=2,
        population_size=3,
        proposals_per_step=2,
        full_eval_every=1,
        max_demos_per_proposal=1,
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


########################################
# Prompt equivalence & type handling   #
########################################


def test_prompts_equivalent_detects_text_and_list_prompts():
    """
    _prompts_equivalent should treat prompts with identical trimmed text
    (or identical LIST messages) as equivalent, and different ones as not.
    """
    cfg = SIMBAConfig()
    runner = SIMBARunner(config=cfg, scoring_adapter=None)

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

    lp1 = Prompt(messages_template=msgs1)
    lp2 = Prompt(messages_template=msgs2)
    lp3 = Prompt(messages_template=msgs3)

    assert runner._prompts_equivalent(lp1, lp2) is True
    assert runner._prompts_equivalent(lp1, lp3) is False


def test_generate_child_prompt_rejects_equivalent_and_type_changes():
    """
    _generate_child_prompt should return None when:
    - the rewriter produces an equivalent prompt, or
    - the rewriter changes the prompt type.
    """

    class EchoRewriter:
        # Returns the old prompt unchanged
        def rewrite(self, **kwargs):
            return kwargs["old_prompt"]

    cfg = SIMBAConfig()
    runner = SIMBARunner(config=cfg, scoring_adapter=None)
    runner._rewriter = EchoRewriter()

    parent = PromptConfiguration.new(
        prompts={SIMBARunner.SINGLE_MODULE_ID: Prompt(text_template="ROOT")}
    )

    child = runner._generate_child_prompt(
        SIMBAStrategy.APPEND_RULE,
        SIMBARunner.SINGLE_MODULE_ID,
        parent,
        feedback_text="some feedback",
        minibatch=[],
    )
    assert child is None  # equivalent => rejected

    # Now a rewriter that changes type: TEXT -> LIST
    class ListRewriter:
        def rewrite(self, **kwargs):
            old_prompt = kwargs["old_prompt"]
            return Prompt(
                messages_template=[
                    PromptMessage(role="user", content=old_prompt.text_template)
                ]
            )

    runner._rewriter = ListRewriter()
    child2 = runner._generate_child_prompt(
        SIMBAStrategy.APPEND_RULE,
        SIMBARunner.SINGLE_MODULE_ID,
        parent,
        feedback_text="some feedback",
        minibatch=[],
    )
    assert child2 is None  # type change => rejected


####################################
# Strategy selection & demo blocks #
####################################


def test_simba_initializes_strategies_based_on_max_demos():
    """
    When max_demos_per_proposal > 0, SIMBA should include both APPEND_DEMO
    and APPEND_RULE; when it is 0, only APPEND_RULE should be active.
    """
    cfg_demo = SIMBAConfig(max_demos_per_proposal=2)
    runner_demo = SIMBARunner(config=cfg_demo, scoring_adapter=None)
    assert set(runner_demo._strategies) == {
        SIMBAStrategy.APPEND_DEMO,
        SIMBAStrategy.APPEND_RULE,
    }

    cfg_rule_only = SIMBAConfig(max_demos_per_proposal=0)
    runner_rule_only = SIMBARunner(config=cfg_rule_only, scoring_adapter=None)
    assert runner_rule_only._strategies == [SIMBAStrategy.APPEND_RULE]


def test_sample_strategy_respects_configured_strategies():
    """
    _sample_strategy should always return APPEND_RULE when demos are disabled,
    and must always return one of the configured strategies otherwise.
    """
    cfg_rule_only = SIMBAConfig(max_demos_per_proposal=0)
    runner_rule_only = SIMBARunner(config=cfg_rule_only, scoring_adapter=None)

    for _ in range(10):
        assert runner_rule_only._sample_strategy() is SIMBAStrategy.APPEND_RULE

    cfg_both = SIMBAConfig(max_demos_per_proposal=3, random_seed=0)
    runner_both = SIMBARunner(config=cfg_both, scoring_adapter=None)
    for _ in range(10):
        s = runner_both._sample_strategy()
        assert s in runner_both._strategies


def test_build_demo_block_uses_golden_shapes_and_truncates():
    """
    _build_demo_block should pull (input, expected) pairs from both Golden
    and ConversationalGolden instances, and respect demo_input_max_chars.
    """
    cfg = SIMBAConfig(
        max_demos_per_proposal=2,
        demo_input_max_chars=5,
    )
    runner = SIMBARunner(config=cfg, scoring_adapter=None)

    g1 = Golden(input="hello world", expected_output="long expected output")
    g2 = ConversationalGolden(
        scenario="greetings from chatgpt",
        expected_outcome="short",
    )

    block = runner._build_demo_block([g1, g2])
    # We should see two demo blocks separated by a blank line
    demos = block.split("\n\n")
    assert len(demos) == 2

    # Truncation to 5 chars
    assert "Input: hello" in demos[0]
    assert "Output: long "[:5] in demos[0]  # prefix truncated

    assert "Input: greet"[:5] in demos[1]
    assert "Output: short"[:5] in demos[1]


def test_build_feedback_for_strategy_append_rule():
    """
    _build_feedback_for_strategy(APPEND_RULE) should embed the strategy
    instructions and the evaluation feedback text.
    """
    cfg = SIMBAConfig()
    runner = SIMBARunner(config=cfg, scoring_adapter=None)

    feedback = "The answers were too verbose."
    text = runner._build_feedback_for_strategy(
        SIMBAStrategy.APPEND_RULE,
        feedback_text=feedback,
        minibatch=[],
    )

    assert "Append a concise natural-language rule" in text
    assert "Evaluation feedback:" in text
    assert "The answers were too verbose." in text


def test_build_feedback_for_strategy_append_demo_includes_demos():
    """
    _build_feedback_for_strategy(APPEND_DEMO) should include both the
    strategy instructions and a demo block built from the minibatch.
    """
    cfg = SIMBAConfig(max_demos_per_proposal=1, demo_input_max_chars=50)
    runner = SIMBARunner(config=cfg, scoring_adapter=None)

    g = Golden(input="question", expected_output="answer")
    text = runner._build_feedback_for_strategy(
        SIMBAStrategy.APPEND_DEMO,
        feedback_text="Be more concise.",
        minibatch=[g],
    )

    assert "Append one or more concrete input/output demonstrations" in text
    assert "Evaluation feedback:" in text
    assert "Be more concise." in text
    assert "Candidate demos built from the current minibatch:" in text
    assert "Input: question" in text
    assert "Output: answer" in text


###########################################
# Progress & error status callback wiring #
###########################################


def test_update_progress_and_error_use_status_callback():
    """
    _update_progress and _update_error should forward structured events to
    the status_callback, allowing PromptOptimizer to drive a progress bar.
    """
    cfg = SIMBAConfig(iterations=2)
    runner = SIMBARunner(config=cfg, scoring_adapter=None)

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
