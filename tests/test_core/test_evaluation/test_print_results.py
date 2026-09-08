"""Regression tests for DisplayConfig.print_results output control."""

import asyncio
import importlib
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.execute._common import _log_gather_timeout
from deepeval.evaluate.inspect_prompt import _should_prompt
from deepeval.metrics import BaseMetric, ExactMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager
from deepeval.utils import (
    should_print_evaluation_output,
    suppress_evaluation_output,
)


class _AlwaysPassMetric(BaseMetric):
    """Deterministic metric that always scores 1.0. No LLM calls."""

    def __init__(self):
        self.threshold = 0.5
        self.strict_mode = False

    @property
    def __name__(self):
        return "AlwaysPass"

    def measure(self, test_case):
        self.success = True
        self.score = 1.0
        return self.score

    async def a_measure(self, test_case):
        return self.measure(test_case)

    def is_successful(self):
        return self.success


class _AlwaysFailMetric(_AlwaysPassMetric):
    def measure(self, test_case, *args, **kwargs):
        raise RuntimeError("simulated metric failure")

    async def a_measure(self, test_case, *args, **kwargs):
        raise RuntimeError("simulated metric failure")


_QUIET_DISPLAY = DisplayConfig(print_results=False)
_QUIET_ASYNC = AsyncConfig(run_async=False)
_evaluate_module = importlib.import_module("deepeval.evaluate.evaluate")
_execute_module = importlib.import_module("deepeval.evaluate.execute")
_inspect_prompt_module = importlib.import_module(
    "deepeval.evaluate.inspect_prompt"
)


def _make_case(label: str) -> LLMTestCase:
    return LLMTestCase(input=f"input-{label}", actual_output=f"output-{label}")


@pytest.fixture(autouse=True)
def _reset_test_run_manager():
    global_test_run_manager.reset()
    yield
    global_test_run_manager.reset()


class TestPrintResults:
    @pytest.mark.parametrize("run_async", [False, True])
    def test_false_suppresses_all_output(self, capsys, run_async):
        display_config = DisplayConfig(print_results=False)
        evaluate(
            test_cases=[_make_case("quiet")],
            metrics=[_AlwaysPassMetric()],
            display_config=display_config,
            async_config=AsyncConfig(run_async=run_async),
        )

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert display_config.show_indicator is True

    def test_true_preserves_default_output(self, capsys):
        evaluate(
            test_cases=[_make_case("visible")],
            metrics=[_AlwaysPassMetric()],
            display_config=DisplayConfig(
                show_indicator=False, print_results=True
            ),
            async_config=_QUIET_ASYNC,
        )

        captured = capsys.readouterr()
        assert "Evaluation completed" in captured.out

    def test_false_overrides_metric_verbose_output(self, capsys):
        metric = ExactMatchMetric(verbose_mode=True)
        test_case = LLMTestCase(
            input="input",
            actual_output="same",
            expected_output="same",
        )
        evaluate(
            test_cases=[test_case],
            metrics=[metric],
            display_config=DisplayConfig(
                print_results=False, verbose_mode=True
            ),
            async_config=_QUIET_ASYNC,
        )

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert metric.verbose_mode is True

        evaluate(
            test_cases=[test_case],
            metrics=[metric],
            display_config=DisplayConfig(show_indicator=False),
            async_config=_QUIET_ASYNC,
        )
        assert "Exact Match Verbose Logs" in capsys.readouterr().out

    def test_false_suppresses_evaluation_owned_logging(self, caplog):
        caplog.set_level(logging.INFO, logger="deepeval.metrics.indicator")

        evaluate(
            test_cases=[_make_case("failing")],
            metrics=[_AlwaysFailMetric()],
            display_config=_QUIET_DISPLAY,
            async_config=AsyncConfig(run_async=True),
            error_config=ErrorConfig(ignore_errors=True),
        )

        indicator_records = [
            record
            for record in caplog.records
            if record.name == "deepeval.metrics.indicator"
        ]
        assert indicator_records == []

    def test_quiet_async_iterator_callback_ignores_caller_context(
        self, caplog, settings
    ):
        """Task callbacks stay quiet after the iterator yields to user code."""
        with settings.edit(persist=False):
            settings.DEEPEVAL_DEBUG_ASYNC = True

        logger_name = "deepeval.evaluate.execute"
        caplog.set_level(logging.INFO, logger=logger_name)
        loop = asyncio.new_event_loop()

        try:
            asyncio.set_event_loop(loop)
            iterator = _execute_module.a_execute_agentic_test_cases_from_loop(
                goldens=[Golden(input="quiet callback")],
                trace_metrics=[_AlwaysPassMetric()],
                test_results=[],
                loop=loop,
                display_config=DisplayConfig(
                    show_indicator=False, print_results=False
                ),
                async_config=AsyncConfig(run_async=True),
                error_config=ErrorConfig(
                    ignore_errors=True, skip_on_missing_params=True
                ),
            )

            golden = next(iterator)
            assert should_print_evaluation_output() is True

            async def failing_app(_):
                raise RuntimeError("quiet callback failure")

            task = asyncio.create_task(failing_app(golden.input))
            try:
                iterator.send(task)
            except StopIteration:
                pass
            for _ in iterator:
                pass

            evaluation_records = [
                record
                for record in caplog.records
                if record.name.startswith(logger_name)
            ]
            assert evaluation_records == []
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_output_context_suppresses_timeout_logging(self, caplog):
        logger = logging.getLogger("deepeval.tests.quiet_timeout")
        caplog.set_level(logging.WARNING, logger=logger.name)

        with suppress_evaluation_output():
            _log_gather_timeout(logger, pending=1)

        assert [
            record for record in caplog.records if record.name == logger.name
        ] == []

        _log_gather_timeout(logger, pending=1)
        assert (
            len(
                [
                    record
                    for record in caplog.records
                    if record.name == logger.name
                ]
            )
            == 1
        )

    def test_false_disables_tty_inspect_prompt(self, tmp_path, monkeypatch):
        saved_path = tmp_path / "run.json"
        saved_path.write_text("{}", encoding="utf-8")
        manager = SimpleNamespace(
            last_saved_path=saved_path,
            test_run=SimpleNamespace(
                test_cases=[SimpleNamespace(trace=object())]
            ),
        )
        monkeypatch.delenv("DEEPEVAL_NO_INSPECT_PROMPT", raising=False)
        monkeypatch.setattr(
            _inspect_prompt_module.sys.stdout, "isatty", lambda: True
        )

        assert _should_prompt(manager, _QUIET_DISPLAY) is False

    def test_metric_collection_quiet_mode_suppresses_confirmation(
        self, monkeypatch, capsys
    ):
        class FakeApi:
            def send_request(self, **kwargs):
                return {}, "https://example.test/evaluations/quiet"

        opened_links = []
        monkeypatch.setattr(_evaluate_module, "Api", FakeApi)
        monkeypatch.setattr(
            _evaluate_module, "open_browser", opened_links.append
        )

        evaluate(
            test_cases=[_make_case("metric-collection")],
            metric_collection="collection-123",
            display_config=_QUIET_DISPLAY,
        )

        assert opened_links == ["https://example.test/evaluations/quiet"]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    @pytest.mark.parametrize("run_async", [False, True])
    def test_evals_iterator_uses_derived_config_without_muting_caller(
        self, monkeypatch, capsys, run_async
    ):
        supplied_config = DisplayConfig(print_results=False, verbose_mode=True)
        metric = ExactMatchMetric(verbose_mode=True)
        seen_configs = []
        internal_output_states = []

        def fake_execute_agentic_test_cases_from_loop(
            *, goldens, trace_metrics, display_config, **kwargs
        ):
            seen_configs.append(display_config)
            for golden in goldens:
                internal_output_states.append(should_print_evaluation_output())
                yield golden
                internal_output_states.append(should_print_evaluation_output())

        execute_name = (
            "a_execute_agentic_test_cases_from_loop"
            if run_async
            else "execute_agentic_test_cases_from_loop"
        )
        monkeypatch.setattr(
            _execute_module,
            execute_name,
            fake_execute_agentic_test_cases_from_loop,
        )
        dataset = EvaluationDataset(goldens=[Golden(input="input")])
        legacy_wrap_calls = []

        def legacy_wrap_up(run_duration, display_table=True):
            legacy_wrap_calls.append(
                (
                    run_duration,
                    display_table,
                    should_print_evaluation_output(),
                )
            )

        with patch.object(
            global_test_run_manager,
            "wrap_up_test_run",
            side_effect=legacy_wrap_up,
        ) as mock_wrap_up:
            iterator = dataset.evals_iterator(
                metrics=[metric],
                display_config=supplied_config,
                async_config=AsyncConfig(run_async=run_async),
            )
            yielded = [next(iterator)]
            assert should_print_evaluation_output() is True
            print("caller output remains visible")
            with pytest.raises(StopIteration):
                next(iterator)

        assert len(yielded) == 1
        assert internal_output_states == [False, False]
        assert seen_configs[0].show_indicator is False
        assert seen_configs[0].verbose_mode is True
        assert supplied_config.show_indicator is True
        assert supplied_config.verbose_mode is True
        assert metric.verbose_mode is True
        _, kwargs = mock_wrap_up.call_args
        assert kwargs == {"display_table": False}
        assert len(legacy_wrap_calls) == 1
        assert legacy_wrap_calls[0][0] >= 0
        assert legacy_wrap_calls[0][1:] == (False, False)
        captured = capsys.readouterr()
        assert captured.out == "caller output remains visible\n"
        assert captured.err == ""

    def test_evals_iterator_suppresses_span_metric_verbose_logs(
        self, monkeypatch, capsys
    ):
        span_metric = ExactMatchMetric(verbose_mode=True)
        test_case = LLMTestCase(
            input="input",
            actual_output="same",
            expected_output="same",
        )

        def fake_execute_agentic_test_cases_from_loop(*, goldens, **kwargs):
            span_metric.measure(test_case, _show_indicator=False)
            yield from goldens

        monkeypatch.setattr(
            _execute_module,
            "execute_agentic_test_cases_from_loop",
            fake_execute_agentic_test_cases_from_loop,
        )
        dataset = EvaluationDataset(goldens=[Golden(input="input")])
        with patch.object(
            global_test_run_manager, "wrap_up_test_run", return_value=None
        ):
            list(
                dataset.evals_iterator(
                    metrics=None,
                    display_config=DisplayConfig(print_results=False),
                    async_config=_QUIET_ASYNC,
                )
            )

        assert span_metric.verbose_mode is True
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
