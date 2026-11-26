from rich.console import Console

from deepeval.test_run import test_run as test_run_module
from deepeval.prompt.api import ModelSettings
from deepeval.test_run.api import (
    ConversationalApiTestCase,
    LLMApiTestCase,
    MetricData,
    TurnApi,
)


def _record_output(monkeypatch) -> Console:
    recorder = Console(record=True, width=200)
    monkeypatch.setattr(test_run_module, "console", recorder)
    monkeypatch.setattr(test_run_module, "print", recorder.print, raising=False)
    return recorder


def test_results_table_renders_markup_like_turns_literally(monkeypatch):
    recorder = _record_output(monkeypatch)

    run = test_run_module.TestRun(
        conversationalTestCases=[
            ConversationalApiTestCase(
                name="convo [bold]1[/bold]",
                success=True,
                metricsData=[],
                turns=[
                    TurnApi(order=0, role="user", content="[/INST] hi"),
                    TurnApi(order=1, role="assistant", content="[red]x[/red]"),
                ],
            )
        ]
    )

    test_run_module.TestRunManager().display_results_table(
        run, test_run_module.TestRunResultDisplay.ALL
    )
    text = recorder.export_text()

    assert "convo [bold]1[/bold]" in text
    assert "[/INST] hi" in text
    assert "[red]x[/red]" in text


def test_results_table_renders_markup_like_names_literally(monkeypatch):
    recorder = _record_output(monkeypatch)

    run = test_run_module.TestRun(
        testCases=[
            LLMApiTestCase(
                name="case [/INST]",
                input="in",
                actualOutput="out",
                success=False,
                metricsData=[
                    MetricData(
                        name="Exact Match [strict]",
                        score=0.0,
                        threshold=0.5,
                        success=False,
                        strictMode=False,
                    )
                ],
            )
        ]
    )

    test_run_module.TestRunManager().display_results_table(
        run, test_run_module.TestRunResultDisplay.ALL
    )
    text = recorder.export_text()

    assert "case [/INST]" in text
    assert "Exact Match [strict]" in text


def test_prompts_panels_render_markup_like_values_literally(monkeypatch):
    recorder = _record_output(monkeypatch)

    test_run_module.TestRunManager()._render_prompts_panels(
        prompts=[
            test_run_module.PromptData(
                alias="p",
                model_settings=ModelSettings(stop_sequence=["</s>", "[/INST]"]),
            ),
            test_run_module.PromptData(alias="alias [/INST]", hash="h[red]1"),
        ]
    )
    text = recorder.export_text()

    assert "['</s>', '[/INST]']" in text
    assert "alias [/INST]" in text
    assert "h[red]1" in text
    assert "None" in text
