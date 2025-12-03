import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimizer.algorithms import MIPROV2
from deepeval.optimizer.types import PromptConfiguration
from deepeval.prompt.prompt import Prompt
from tests.test_core.stubs import StubScoringAdapter, SuffixRewriter


def _make_runner(
    *,
    iterations: int = 3,
    exploration_probability: float = 0.0,
    full_eval_every=None,
) -> MIPROV2:
    """
    Helper that constructs a MIPROV2 runner wired with the shared test stubs.

    - StubScoringAdapter: scores prompts with "CHILD" higher than the root.
    - SuffixRewriter: appends " CHILD" to the prompt text.
    """
    scorer = StubScoringAdapter()
    runner = MIPROV2(
        iterations=iterations,
        exploration_probability=exploration_probability,
        full_eval_every=full_eval_every,
        scorer=scorer,
    )

    # Attach a deterministic rewriter that always produces a strictly "better"
    # child by appending " CHILD" to the text template.
    runner._rewriter = SuffixRewriter(suffix=" CHILD")

    return runner


def test_execute_requires_at_least_one_golden():
    """MIPROV2.execute must reject an empty golden list."""
    runner = _make_runner()
    prompt = Prompt(text_template="ROOT")

    with pytest.raises(DeepEvalError):
        runner.execute(prompt=prompt, goldens=[])


@pytest.mark.asyncio
async def test_a_execute_requires_at_least_one_golden():
    """MIPROV2.a_execute must reject an empty golden list."""
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
        exploration_probability=0.0,
        full_eval_every=None,
    )

    # Swap in the shared StubScoringAdapter explicitly so we can inspect calls.
    scorer = StubScoringAdapter()
    runner.scorer = scorer

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
    assert scorer.a_score_calls, "expected async minibatch_score calls"
    assert scorer.a_pareto_calls, "expected async score_on_pareto calls"
    assert scorer.a_feedback_calls, "expected async feedback calls"


def test_draw_minibatch_respects_minibatch_size():
    """
    _draw_minibatch should use minibatch_size, clamped by the number of goldens.
    """
    runner = MIPROV2(minibatch_size=3, scorer=StubScoringAdapter())

    goldens = list(range(10))
    batch = runner._draw_minibatch(goldens)
    assert len(batch) == 3
    # We sample *with* replacement, so duplicates are allowed, but all
    # elements must be drawn from the original collection.
    assert all(item in goldens for item in batch)


def test_draw_minibatch_clamps_to_available_data():
    """
    When minibatch_size exceeds available goldens, clamp to the number of goldens.
    """
    runner = MIPROV2(minibatch_size=10, scorer=StubScoringAdapter())

    # If we have fewer goldens than minibatch_size, we clamp to n.
    goldens_small = list(range(2))
    batch_small = runner._draw_minibatch(goldens_small)
    assert len(batch_small) == 2


def test_select_candidate_prefers_best_by_minibatch_when_eps_zero():
    """
    With exploration_probability = 0.0, _select_candidate should always pick
    the candidate with the highest mean minibatch score.
    """
    runner = MIPROV2(
        iterations=1,
        exploration_probability=0.0,
        scorer=StubScoringAdapter(),
    )

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
    runner = MIPROV2(scorer=StubScoringAdapter())

    p1 = Prompt(text_template="  Do the thing.  ")
    p2 = Prompt(text_template="Do the thing.")
    p3 = Prompt(text_template="Do a different thing.")

    assert runner._prompts_equivalent(p1, p2)
    assert not runner._prompts_equivalent(p1, p3)
