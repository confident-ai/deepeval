import pytest

from tests.test_core.stubs import (
    StubScoringAdapter,
    SuffixRewriter,
    _DummyRewriter,
)
from deepeval.errors import DeepEvalError
from deepeval.optimization.gepa.configs import GEPAConfig
from deepeval.optimization.gepa.loop import GEPARunner
from deepeval.optimization.types import PromptConfiguration, RunnerStatusType
from deepeval.prompt.prompt import Prompt


##########################
# execute / a_execute    #
##########################


def test_execute_requires_at_least_two_goldens() -> None:
    config = GEPAConfig(iterations=1, minibatch_size=1, pareto_size=1)
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())
    prompt = Prompt(text_template="base")

    with pytest.raises(DeepEvalError, match="requires at least 2 goldens"):
        runner.execute(prompt=prompt, goldens=[object()])


@pytest.mark.asyncio
async def test_a_execute_requires_at_least_two_goldens() -> None:
    config = GEPAConfig(iterations=1, minibatch_size=1, pareto_size=1)
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())
    prompt = Prompt(text_template="base")

    with pytest.raises(DeepEvalError, match="requires at least 2 goldens"):
        await runner.a_execute(prompt=prompt, goldens=[object()])


def test_execute_raises_without_scoring_adapter() -> None:
    config = GEPAConfig(iterations=1, minibatch_size=1, pareto_size=1)
    runner = GEPARunner(config=config, scoring_adapter=None)
    prompt = Prompt(text_template="base")
    goldens = [object(), object()]

    with pytest.raises(DeepEvalError, match="requires a `scoring_adapter`"):
        runner.execute(prompt=prompt, goldens=goldens)


def test_execute_end_to_end_accepts_improved_child_prompt() -> None:
    """
    Full GEPA run with a stub scoring adapter and rewriter:

    - root prompt scores lower on Pareto than the child
    - child is accepted
    - the returned best prompt is the rewritten child
    """
    config = GEPAConfig(
        iterations=1,
        minibatch_size=1,
        pareto_size=1,
        random_seed=0,
    )
    scoring = StubScoringAdapter()
    runner = GEPARunner(config=config, scoring_adapter=scoring)

    # Use a deterministic rewriter that always improves the text.
    runner._rewriter = SuffixRewriter(" CHILD")

    prompt = Prompt(text_template="base")
    goldens = [object(), object()]

    best_prompt, report = runner.execute(prompt=prompt, goldens=goldens)

    assert isinstance(best_prompt, Prompt)
    assert best_prompt.text_template == "base CHILD"

    # Report should be the runtime dict payload
    assert isinstance(report, dict)

    # Reasonable sanity checks on the report payload
    assert set(report.keys()) >= {
        "optimization_id",
        "best_id",
        "accepted_iterations",
        "pareto_scores",
        "parents",
        "prompt_configurations",
    }

    assert len(report["accepted_iterations"]) == 1

    # prompt_configurations should contain at least the best config id
    prompt_cfgs = report["prompt_configurations"]
    assert isinstance(prompt_cfgs, dict)
    assert report["best_id"] in prompt_cfgs


@pytest.mark.asyncio
async def test_a_execute_end_to_end_accepts_improved_child_prompt() -> None:
    """
    Async variant of the full GEPA run using the same stubs.
    """
    config = GEPAConfig(
        iterations=1,
        minibatch_size=1,
        pareto_size=1,
        random_seed=0,
    )
    scoring = StubScoringAdapter()
    runner = GEPARunner(config=config, scoring_adapter=scoring)
    runner._rewriter = SuffixRewriter(" CHILD")

    prompt = Prompt(text_template="base")
    goldens = [object(), object()]

    best_prompt, report = await runner.a_execute(prompt=prompt, goldens=goldens)

    assert isinstance(best_prompt, Prompt)
    assert best_prompt.text_template == "base CHILD"

    assert isinstance(report, dict)
    assert set(report.keys()) >= {
        "optimization_id",
        "best_id",
        "accepted_iterations",
        "pareto_scores",
        "parents",
        "prompt_configurations",
    }

    prompt_cfgs = report["prompt_configurations"]
    assert isinstance(prompt_cfgs, dict)
    assert report["best_id"] in prompt_cfgs


##########################
# Minibatch / acceptance #
##########################


def test_draw_minibatch_respects_fixed_minibatch_size() -> None:
    config = GEPAConfig(
        iterations=1,
        minibatch_size=3,
        minibatch_min_size=1,
        minibatch_max_size=10,
        pareto_size=1,
        random_seed=0,
    )
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())
    d_feedback = list(range(10))

    batch = runner._draw_minibatch(d_feedback)

    assert len(batch) == 3
    assert all(item in d_feedback for item in batch)


def test_draw_minibatch_dynamic_size_within_bounds() -> None:
    config = GEPAConfig(
        iterations=1,
        minibatch_size=None,
        minibatch_min_size=4,
        minibatch_max_size=8,
        minibatch_ratio=0.05,
        pareto_size=1,
        random_seed=0,
    )
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())

    d_feedback_large = list(range(100))
    batch_large = runner._draw_minibatch(d_feedback_large)
    # 5% of 100 => 5, clamped between min=4 and max=8
    assert len(batch_large) == 5

    d_feedback_small = list(range(3))
    batch_small = runner._draw_minibatch(d_feedback_small)
    # With only 3 feedback items we should never request more than 3
    assert len(batch_small) == 3


def test_should_accept_child_respects_min_delta_and_jitter() -> None:
    # min_delta = 0.0 -> jitter (1e-6) still applies
    runner = GEPARunner(
        config=GEPAConfig(min_delta=0.0), scoring_adapter=StubScoringAdapter()
    )

    assert runner._should_accept_child(1.0, 1.0) is False
    assert runner._should_accept_child(1.0, 1.0 + 1e-7) is False
    assert runner._should_accept_child(1.0, 1.0 + 2e-6) is True

    # Larger explicit min_delta dominates jitter
    runner2 = GEPARunner(
        config=GEPAConfig(min_delta=0.1),
        scoring_adapter=StubScoringAdapter(),
    )

    assert runner2._should_accept_child(0.5, 0.5 + 0.05) is False
    assert runner2._should_accept_child(0.5, 0.5 + 0.100001) is True


######################################
# Rewriter integration / child build #
######################################


def _make_prompt_config(text: str) -> PromptConfiguration:
    return PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: Prompt(text_template=text)}
    )


def test_generate_child_prompt_returns_none_when_text_unchanged() -> None:
    runner = GEPARunner(
        config=GEPAConfig(), scoring_adapter=StubScoringAdapter()
    )
    parent = _make_prompt_config("  Hello ")
    runner._rewriter = _DummyRewriter()

    child = runner._generate_child_prompt(
        GEPARunner.SINGLE_MODULE_ID, parent, feedback_text="unused"
    )
    assert child is None


def test_generate_child_prompt_returns_new_prompt_when_text_changes() -> None:
    runner = GEPARunner(
        config=GEPAConfig(), scoring_adapter=StubScoringAdapter()
    )
    parent = _make_prompt_config("Hello")
    runner._rewriter = SuffixRewriter(" CHILD")

    child = runner._generate_child_prompt(
        GEPARunner.SINGLE_MODULE_ID, parent, feedback_text="unused"
    )
    assert isinstance(child, Prompt)
    assert child.text_template == "Hello CHILD"


@pytest.mark.asyncio
async def test_a_generate_child_prompt_async_mirrors_sync_behavior() -> None:
    runner = GEPARunner(
        config=GEPAConfig(), scoring_adapter=StubScoringAdapter()
    )
    parent = _make_prompt_config("Hello")
    runner._rewriter = SuffixRewriter(" CHILD")

    child = await runner._a_generate_child_prompt(
        GEPARunner.SINGLE_MODULE_ID, parent, feedback_text="unused"
    )
    assert isinstance(child, Prompt)
    assert child.text_template == "Hello CHILD"


def test_make_child_clones_parent_and_sets_parent_id() -> None:
    runner = GEPARunner(
        config=GEPAConfig(), scoring_adapter=StubScoringAdapter()
    )
    parent_prompt = Prompt(text_template="root")
    parent_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: parent_prompt}
    )

    child_prompt = Prompt(text_template="child")
    child_conf = runner._make_child(
        GEPARunner.SINGLE_MODULE_ID, parent_conf, child_prompt
    )

    assert child_conf.parent == parent_conf.id
    assert child_conf.prompts[GEPARunner.SINGLE_MODULE_ID] is child_prompt
    # Ensure parent prompt remains unchanged
    assert parent_conf.prompts[GEPARunner.SINGLE_MODULE_ID] is parent_prompt


def test_accept_child_updates_state_and_returns_iteration_dict() -> None:
    config = GEPAConfig()
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())

    parent_prompt = Prompt(text_template="root")
    child_prompt = Prompt(text_template="root CHILD")

    parent_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: parent_prompt}
    )
    runner._add_prompt_configuration(parent_conf)

    child_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: child_prompt},
        parent=parent_conf.id,
    )

    d_pareto = [object(), object()]

    accepted = runner._accept_child(
        GEPARunner.SINGLE_MODULE_ID,
        parent_conf,
        child_conf,
        d_pareto,
        parent_score=0.5,
        child_score=1.0,
    )

    # Child must be registered with a Pareto score
    assert child_conf.id in runner.pareto_score_table
    assert isinstance(accepted, dict)
    assert accepted["parent"] == parent_conf.id
    assert accepted["child"] == child_conf.id
    assert accepted["module"] == GEPARunner.SINGLE_MODULE_ID
    assert accepted["before"] == pytest.approx(0.5)
    assert accepted["after"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_a_accept_child_updates_state_and_returns_iteration_dict() -> (
    None
):
    config = GEPAConfig()
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())

    parent_prompt = Prompt(text_template="root")
    child_prompt = Prompt(text_template="root CHILD")

    parent_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: parent_prompt}
    )
    runner._add_prompt_configuration(parent_conf)

    child_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: child_prompt},
        parent=parent_conf.id,
    )

    d_pareto = [object(), object()]

    accepted = await runner._a_accept_child(
        GEPARunner.SINGLE_MODULE_ID,
        parent_conf,
        child_conf,
        d_pareto,
        parent_score=0.5,
        child_score=1.0,
    )

    assert child_conf.id in runner.pareto_score_table
    assert isinstance(accepted, dict)
    assert accepted["parent"] == parent_conf.id
    assert accepted["child"] == child_conf.id


#####################################
# Aggregation / tie-breaker / loop  #
#####################################


def test_best_by_aggregate_prefers_child_and_emits_tie_status() -> None:
    """
    _best_by_aggregate should:
      - use the configured tie_breaker (default PREFER_CHILD)
      - emit a TIE status when multiple configs share the best total
    """
    config = GEPAConfig()
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())

    root_prompt = Prompt(text_template="root")
    child_prompt = Prompt(text_template="root CHILD")

    root_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: root_prompt}
    )
    child_conf = PromptConfiguration.new(
        prompts={GEPARunner.SINGLE_MODULE_ID: child_prompt},
        parent=root_conf.id,
    )

    runner._add_prompt_configuration(root_conf)
    runner._add_prompt_configuration(child_conf)

    # Equal aggregate scores to force a tie
    runner.pareto_score_table = {
        root_conf.id: [1.0],
        child_conf.id: [1.0],
    }

    events = []

    def status_cb(kind, *, detail, step_index=None, total_steps=None):
        events.append((kind, detail, step_index, total_steps))

    runner.status_callback = status_cb

    best = runner._best_by_aggregate()

    # With PREFER_CHILD, the non root config should be chosen
    assert best.id == child_conf.id

    tie_events = [e for e in events if e[0] is RunnerStatusType.TIE]
    assert tie_events, "Expected at least one TIE status callback"
    tie_detail = tie_events[0][1]
    assert "tie on aggregate" in tie_detail
    assert "using tie_breaker='prefer_child'" in tie_detail


def test_run_loop_iteration_reports_progress_and_stops_on_false() -> None:
    """
    _run_loop_iteration should:
      - emit an initial PROGRESS event at iteration 0
      - emit PROGRESS per successful iteration
      - stop when the iteration callback returns False
    """
    config = GEPAConfig(iterations=3)
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())

    events = []

    def status_cb(kind, *, detail, step_index=None, total_steps=None):
        events.append((kind, step_index, total_steps, detail))

    runner.status_callback = status_cb

    calls = {"count": 0}

    def gepa_iteration() -> bool:
        calls["count"] += 1
        # stop after the second call returns False
        return calls["count"] < 2

    runner._run_loop_iteration(gepa_iteration)

    # Initial progress event at step_index=0 plus one successful iteration.
    progress_events = [e for e in events if e[0] is RunnerStatusType.PROGRESS]
    assert len(progress_events) == 2
    # First call should be iteration 0
    assert progress_events[0][1] == 0
    assert progress_events[0][2] == config.iterations


@pytest.mark.asyncio
async def test_a_run_loop_iteration_reports_error_and_stops() -> None:
    """
    _a_run_loop_iteration should:
      - emit initial PROGRESS
      - emit ERROR on exception
      - stop without propagating the exception
    """
    config = GEPAConfig(iterations=3)
    runner = GEPARunner(config=config, scoring_adapter=StubScoringAdapter())

    events = []

    def status_cb(kind, *, detail, step_index=None, total_steps=None):
        events.append((kind, step_index, total_steps, detail))

    runner.status_callback = status_cb

    async def failing_iteration() -> bool:
        raise ValueError("boom")

    # Should not raise, but should report an ERROR
    await runner._a_run_loop_iteration(failing_iteration)

    kinds = [e[0] for e in events]
    assert kinds[0] is RunnerStatusType.PROGRESS  # initial event
    assert any(k is RunnerStatusType.ERROR for k in kinds)
    error_events = [e for e in events if e[0] is RunnerStatusType.ERROR]
    assert "boom" in error_events[0][3]
