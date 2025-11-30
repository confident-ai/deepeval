import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimization.miprov2.configs import MIPROConfig
from deepeval.optimization.miprov2.loop import MIPRORunner
from deepeval.optimization.types import PromptConfiguration
from deepeval.prompt.prompt import Prompt
from tests.test_core.stubs import StubScoringAdapter, SuffixRewriter


def _make_runner(
    *,
    iterations: int = 3,
    min_delta: float = 0.0,
    exploration_probability: float = 0.0,
    full_eval_every=None,
) -> MIPRORunner:
    """
    Helper that constructs a MIPRORunner wired with the shared test stubs.

    - StubScoringAdapter: scores prompts with "CHILD" higher than the root.
    - SuffixRewriter: appends " CHILD" to the prompt text.
    """
    config = MIPROConfig(
        iterations=iterations,
        min_delta=min_delta,
        exploration_probability=exploration_probability,
        full_eval_every=full_eval_every,
    )
    scoring_adapter = StubScoringAdapter()
    runner = MIPRORunner(config=config, scoring_adapter=scoring_adapter)

    # Attach a deterministic rewriter that always produces a strictly “better”
    # child by appending " CHILD" to the text template.
    runner._rewriter = SuffixRewriter(suffix=" CHILD")

    return runner


def test_execute_requires_at_least_one_golden():
    """MIPRORunner.execute must reject an empty golden list."""
    runner = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        runner.execute(prompt=prompt, goldens=[])


@pytest.mark.asyncio
async def test_a_execute_requires_at_least_one_golden():
    """MIPRORunner.a_execute must reject an empty golden list."""
    runner = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        await runner.a_execute(prompt=prompt, goldens=[])


def test_execute_accepts_improving_children_and_returns_best():
    """
    Sync path: with StubScoringAdapter + SuffixRewriter, the optimizer
    should end up with a best prompt whose text contains 'CHILD'.
    """
    runner = _make_runner(
        iterations=4,
        min_delta=0.0,
        exploration_probability=0.0,
        full_eval_every=None,  # rely on final full_eval at the end
    )

    prompt = Prompt(text_template="ROOT")
    # Goldens can be any opaque objects; StubScoringAdapter only stores them.
    goldens = ["g1", "g2", "g3", "g4"]

    best_prompt, report = runner.execute(prompt=prompt, goldens=goldens)

    assert "CHILD" in (best_prompt.text_template or "")
    # Ensure we recorded a full-eval score for the returned best id.
    assert report["best_id"] in report["pareto_scores"]


@pytest.mark.asyncio
async def test_a_execute_uses_async_paths_and_accepts_children():
    """
    Async path: verify that the async scoring + feedback methods are used and
    that we still prefer CHILD prompts.
    """
    runner = _make_runner(
        iterations=4,
        min_delta=0.0,
        exploration_probability=0.0,
        full_eval_every=None,
    )

    # Swap in the shared StubScoringAdapter explicitly so we can inspect calls.
    scoring_adapter = StubScoringAdapter()
    runner.scoring_adapter = scoring_adapter

    # Re-attach the SuffixRewriter to guarantee an improving child.
    runner._rewriter = SuffixRewriter(suffix=" CHILD")

    prompt = Prompt(text_template="ROOT")
    goldens = ["g1", "g2", "g3"]

    best_prompt, report = await runner.a_execute(
        prompt=prompt,
        goldens=goldens,
    )

    assert "CHILD" in (best_prompt.text_template or "")
    assert report["best_id"] in report["pareto_scores"]

    # Confirm that async scoring/feedback paths were actually exercised.
    assert scoring_adapter.a_score_calls, "expected async minibatch_score calls"
    assert (
        scoring_adapter.a_pareto_calls
    ), "expected async score_on_pareto calls"
    assert scoring_adapter.a_feedback_calls, "expected async feedback calls"


def test_draw_minibatch_with_dynamic_ratio_respects_bounds():
    """
    _draw_minibatch should respect minibatch_min_size/minibatch_max_size and
    the dynamic ratio, clamping the effective size between these bounds and
    the number of available goldens.
    """
    cfg = MIPROConfig(
        minibatch_size=None,
        minibatch_ratio=0.1,
        minibatch_min_size=4,
        minibatch_max_size=16,
    )
    runner = MIPRORunner(config=cfg, scoring_adapter=StubScoringAdapter())

    goldens = list(range(50))
    batch = runner._draw_minibatch(goldens)

    assert 4 <= len(batch) <= 16
    # We sample *with* replacement, so duplicates are allowed, but all
    # elements must be drawn from the original collection.
    assert all(item in goldens for item in batch)


def test_draw_minibatch_respects_fixed_minibatch_size():
    """
    When minibatch_size is set, _draw_minibatch should use that value,
    clamped by the number of goldens.
    """
    cfg = MIPROConfig(
        minibatch_size=3,
        minibatch_ratio=0.5,  # ignored
        minibatch_min_size=1,
        minibatch_max_size=10,
    )
    runner = MIPRORunner(config=cfg, scoring_adapter=StubScoringAdapter())

    goldens = list(range(10))
    batch = runner._draw_minibatch(goldens)
    assert len(batch) == 3

    # If we have fewer goldens than minibatch_size, we clamp to n.
    goldens_small = list(range(2))
    batch_small = runner._draw_minibatch(goldens_small)
    assert len(batch_small) == 2


def test_select_candidate_prefers_best_by_minibatch_when_eps_zero():
    """
    With exploration_probability = 0.0, _select_candidate should always pick
    the candidate with the highest mean minibatch score.
    """
    cfg = MIPROConfig(
        iterations=1,
        exploration_probability=0.0,
    )
    runner = MIPRORunner(config=cfg, scoring_adapter=StubScoringAdapter())

    # Two candidates: one "ROOT", one "ROOT CHILD" (better).
    root_prompt = Prompt(text_template="ROOT")
    child_prompt = Prompt(text_template="ROOT CHILD")

    root_pc = PromptConfiguration.new(
        prompts={runner.SINGLE_MODULE_ID: root_prompt}
    )
    child_pc = PromptConfiguration.new(
        prompts={runner.SINGLE_MODULE_ID: child_prompt}
    )

    runner._add_prompt_configuration(root_pc)
    runner._add_prompt_configuration(child_pc)

    # Explicitly set the minibatch means: child > root.
    runner._record_minibatch_score(root_pc.id, 0.5)
    runner._record_minibatch_score(child_pc.id, 1.0)

    chosen = runner._select_candidate()
    assert chosen.id == child_pc.id


def test_prompts_equivalent_for_text_templates():
    """
    _prompts_equivalent should ignore leading/trailing whitespace for TEXT
    prompts and treat identical normalized texts as equivalent.
    """
    cfg = MIPROConfig()
    runner = MIPRORunner(config=cfg, scoring_adapter=StubScoringAdapter())

    p1 = Prompt(text_template="  Do the thing.  ")
    p2 = Prompt(text_template="Do the thing.")
    p3 = Prompt(text_template="Do a different thing.")

    assert runner._prompts_equivalent(p1, p2)
    assert not runner._prompts_equivalent(p1, p3)
