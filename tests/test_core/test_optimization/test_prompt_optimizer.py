import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimization.configs import OptimizerDisplayConfig
from deepeval.optimization.prompt_optimizer import PromptOptimizer
from deepeval.optimization.adapters.deepeval_scoring_adapter import (
    DeepEvalScoringAdapter,
)
from deepeval.optimization.gepa.loop import GEPARunner
from deepeval.optimization.types import (
    OptimizationReport,
    RunnerStatusType,
)
from deepeval.prompt.prompt import Prompt
from tests.test_core.stubs import (
    _DummyMetric,
    AsyncDummyRunner,
    DummyProgress,
    DummyRunner,
    DummyRunnerForOptimize,
    SyncDummyRunner,
)


##############################
# _build_default_* and wiring
##############################


def _dummy_model_callback(**_kwargs):
    return "ok"


def test_build_default_scoring_adapter_requires_metrics():
    with pytest.raises(DeepEvalError, match="requires a `metrics`"):
        PromptOptimizer(
            model_callback=_dummy_model_callback,
            metrics=None,
            display_config=OptimizerDisplayConfig(show_indicator=False),
        )


def test_build_default_runner_constructs_gepa_runner_and_sets_callbacks():
    metric = _DummyMetric()
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[metric],
        display_config=OptimizerDisplayConfig(show_indicator=False),
        algorithm="gepa",
    )

    runner = optimizer._build_default_runner()

    assert isinstance(runner, GEPARunner)

    # scoring adapter should be a DeepEvalScoringAdapter with our metric
    scoring_adapter = runner.scoring_adapter
    assert isinstance(scoring_adapter, DeepEvalScoringAdapter)
    assert getattr(scoring_adapter, "metrics", None) is not None
    assert any(isinstance(m, _DummyMetric) for m in scoring_adapter.metrics)

    # callbacks must be wired to the optimizer; status_callback is a bound method
    cb = runner.status_callback
    assert callable(cb)
    # bound to this optimizer instance
    assert getattr(cb, "__self__", None) is optimizer
    # and it wraps PromptOptimizer._on_status
    assert getattr(cb, "__func__", None) is PromptOptimizer._on_status

    # model_callback should be the same callable we passed in
    assert runner.model_callback is optimizer.model_callback


def test_build_default_runner_unsupported_algorithm_raises():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
        algorithm="not-gepa",
    )

    with pytest.raises(
        DeepEvalError, match="Unsupported optimization algorithm"
    ):
        optimizer._build_default_runner()


def test_set_runner_wires_callbacks():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
    )

    runner = DummyRunner()
    optimizer.set_runner(runner)

    assert optimizer.runner is runner
    assert runner.model_callback is optimizer.model_callback

    cb = runner.status_callback
    assert callable(cb)
    assert getattr(cb, "__self__", None) is optimizer
    assert getattr(cb, "__func__", None) is PromptOptimizer._on_status


#####################################
# optimize + _run_optimization paths
#####################################


def test_optimize_with_custom_runner_attaches_report_and_returns_prompt():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
    )
    # Ensure optimize() uses the synchronous execute() path
    optimizer.async_config.run_async = False

    runner = DummyRunnerForOptimize()
    optimizer.set_runner(runner)

    original_prompt = Prompt(text_template="base")
    goldens = [object(), object()]

    best = optimizer.optimize(prompt=original_prompt, goldens=goldens)

    # Returned prompt should be the optimized one from the runner
    assert isinstance(best, Prompt)
    assert best.text_template == "optimized"

    # Runner should have seen the original prompt and goldens
    used_prompt, used_goldens = runner.last_execute_args
    assert used_prompt is original_prompt
    assert used_goldens is goldens

    # OptimizationReport should be attached
    report = getattr(best, "optimization_report", None)
    assert isinstance(report, OptimizationReport)
    assert report.optimization_id == "opt-123"
    assert report.best_id == "best"


def test_run_optimization_uses_sync_execute_when_run_async_false():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
    )
    runner = SyncDummyRunner()
    optimizer.runner = runner
    # Ensure sync path
    optimizer.async_config.run_async = False

    prompt = Prompt(text_template="base")
    goldens = [object()]

    best_prompt, report = optimizer._run_optimization(
        prompt=prompt, goldens=goldens
    )

    assert runner.execute_calls == 1
    assert runner.a_execute_calls == 0
    assert best_prompt is prompt
    assert report["optimization_id"] == "sync-id"


def test_run_optimization_uses_async_execute_when_run_async_true():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
    )
    runner = AsyncDummyRunner()
    optimizer.runner = runner
    optimizer.async_config.run_async = True

    prompt = Prompt(text_template="base")
    goldens = [object()]

    best_prompt, report = optimizer._run_optimization(
        prompt=prompt, goldens=goldens
    )

    assert runner.a_execute_calls == 1
    assert best_prompt is prompt
    assert report["optimization_id"] == "opt-async"


##################
# _on_status()
##################


def test_on_status_error_prints_message_when_indicator_disabled(capsys):
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
    )

    optimizer._on_status(
        RunnerStatusType.ERROR,
        detail="something went wrong",
        step_index=None,
        total_steps=None,
    )

    out = capsys.readouterr().out.strip()
    assert out == "[GEPA] something went wrong"


def test_on_status_tie_respects_announce_ties_flag(capsys):
    # Ties disabled: no output
    opt_quiet = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(
            show_indicator=False, announce_ties=False
        ),
    )
    opt_quiet._on_status(
        RunnerStatusType.TIE,
        detail="tie detail",
        step_index=None,
        total_steps=None,
    )
    out_quiet = capsys.readouterr().out
    assert out_quiet == ""

    # Ties enabled: one-line message
    opt_verbose = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(
            show_indicator=False, announce_ties=True
        ),
    )
    opt_verbose._on_status(
        RunnerStatusType.TIE,
        detail="tie detail",
        step_index=None,
        total_steps=None,
    )
    out_verbose = capsys.readouterr().out.strip()
    assert out_verbose == "[GEPA] tie detail"


def test_on_status_progress_updates_progress_when_indicator_enabled():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=True),
    )

    progress = DummyProgress()
    task_id = 42
    optimizer._progress_state = (progress, task_id)

    optimizer._on_status(
        RunnerStatusType.PROGRESS,
        detail="• iteration 1/5",
        step_index=1,
        total_steps=5,
    )

    # We expect at least an update(total), an advance, and an update(description)
    kinds = [k for (k, _, _) in progress.records]
    assert "update" in kinds
    assert "advance" in kinds

    total_updates = [
        kwargs
        for kind, _, kwargs in progress.records
        if kind == "update" and "total" in kwargs
    ]
    assert total_updates and total_updates[0]["total"] == 5

    desc_updates = [
        kwargs
        for kind, _, kwargs in progress.records
        if kind == "update" and "description" in kwargs
    ]
    assert desc_updates
    desc = desc_updates[-1]["description"]
    assert "Optimizing prompt with GEPA" in desc
    assert "[rgb(25,227,160)]" in desc


def test_format_progress_description_includes_colored_detail():
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=OptimizerDisplayConfig(show_indicator=False),
    )

    text = optimizer._format_progress_description("details here")
    assert (
        text == "Optimizing prompt with GEPA [rgb(25,227,160)]details here[/]"
    )
