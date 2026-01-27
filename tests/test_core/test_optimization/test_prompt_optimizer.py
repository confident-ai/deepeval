import pytest
import os
from deepeval.errors import DeepEvalError
from deepeval.optimizer.configs import DisplayConfig
from deepeval.optimizer.prompt_optimizer import PromptOptimizer
from deepeval.optimizer.types import (
    RunnerStatusType,
)
from tests.test_core.stubs import (
    _DummyMetric,
    DummyProgress,
)

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


##############################
# Validation tests           #
##############################


def _dummy_model_callback(**_kwargs):
    return "ok"


def test_build_default_scorer_requires_metrics():
    with pytest.raises(DeepEvalError, match="requires a `metrics`"):
        PromptOptimizer(
            model_callback=_dummy_model_callback,
            metrics=None,
            display_config=DisplayConfig(show_indicator=False),
        )


def test_build_default_scorer_rejects_non_metric_types():
    # metrics must be BaseMetric, BaseConversationalMetric subclasses
    with pytest.raises(
        DeepEvalError,
        match="expected all elements of `metrics`",
    ):
        PromptOptimizer(
            model_callback=_dummy_model_callback,
            metrics=[object()],
            display_config=DisplayConfig(show_indicator=False),
        )


##################
# _on_status()
##################


def test_on_status_error_prints_message_when_indicator_disabled(capsys):
    optimizer = PromptOptimizer(
        model_callback=_dummy_model_callback,
        metrics=[_DummyMetric()],
        display_config=DisplayConfig(show_indicator=False),
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
        display_config=DisplayConfig(show_indicator=False, announce_ties=False),
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
        display_config=DisplayConfig(show_indicator=False, announce_ties=True),
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
        display_config=DisplayConfig(show_indicator=True),
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
        display_config=DisplayConfig(show_indicator=False),
    )

    text = optimizer._format_progress_description("details here")
    assert (
        text == "Optimizing prompt with GEPA [rgb(25,227,160)]details here[/]"
    )
