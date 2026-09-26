"""Exercise the public iterator with deterministic component and trace metrics."""

import asyncio

import pytest

from deepeval.metrics import BaseMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, CacheConfig
from deepeval.tracing.context import (
    next_agent_span,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.tracing import trace_manager


class RecordingMetric(BaseMetric):
    threshold = 1
    requires_trace = True
    seen = []

    def measure(self, test_case, *args, **kwargs):
        type(self).seen.append((test_case.input, test_case._trace_dict))
        self.score = 1
        self.reason = "Captured"
        self.success = True
        return 1

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Recording"


@pytest.mark.parametrize(
    "pipeline",
    ["strands", "agentcore", "openinference", "pydantic_ai"],
    indirect=True,
)
@pytest.mark.parametrize(
    "run_async,schedule", [(False, False), (True, False), (True, True)]
)
def test_public_iterator_scores_complete_trees(
    pipeline, monkeypatch, run_async, schedule
):
    from deepeval.test_run import global_test_run_manager

    monkeypatch.setattr(
        global_test_run_manager, "save_test_run", lambda *a, **k: None
    )
    monkeypatch.setattr(
        global_test_run_manager, "wrap_up_test_run", lambda *a, **k: None
    )
    import importlib

    inspect_prompt = importlib.import_module("deepeval.evaluate.inspect_prompt")
    monkeypatch.setattr(
        inspect_prompt, "maybe_offer_inspect_tui", lambda *a, **k: None
    )
    RecordingMetric.seen = []
    tracer, processor, _, _ = pipeline
    dataset = EvaluationDataset(
        goldens=[Golden(input="first"), Golden(input="second")]
    )

    def work(input):
        with next_agent_span(metrics=[RecordingMetric()]):
            with tracer.start_as_current_span(
                input,
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                    "openinference.span.kind": "AGENT",
                    "gen_ai.agent.name": input,
                },
            ):
                update_current_span(input=input, output="done")
                update_current_trace(output="done")
                with tracer.start_as_current_span(
                    "tool",
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "openinference.span.kind": "TOOL",
                        "gen_ai.tool.name": "lookup",
                    },
                ):
                    update_current_span(input=input, output="done")

    async def async_work(input):
        await asyncio.sleep(0)
        work(input)

    for golden in dataset.evals_iterator(
        metrics=[RecordingMetric()],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(use_cache=False, write_cache=False),
    ):
        if schedule:
            dataset.evaluate(asyncio.create_task(async_work(golden.input)))
        else:
            work(golden.input)
    assert len(RecordingMetric.seen) == 4  # two trace + two component scores
    assert sorted(input for input, _ in RecordingMetric.seen) == [
        "first",
        "first",
        "second",
        "second",
    ]
    for _, tree in RecordingMetric.seen:
        assert "lookup" in str(tree)
    assert not trace_manager.is_evaluating
    assert not processor._capture.bindings
    processor._otlp_processor.on_end.assert_not_called()
