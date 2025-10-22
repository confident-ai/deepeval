import asyncio
import threading
import pytest
import time
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace_manager
from deepeval.tracing.otel.test_exporter import test_exporter

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
    ]
)

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    instrument=ConfidentInstrumentationSettings(
        is_test_mode=True,
        agent_metrics=[AnswerRelevancyMetric()],
    ),
)


def monitor_list_size(list_to_monitor, max_size_tracker, stop_event):
    """Continuously monitor a list and track its maximum size using threading."""
    while not stop_event.is_set():
        current_size = len(list_to_monitor)
        if current_size > max_size_tracker["max"]:
            max_size_tracker["max"] = current_size
        time.sleep(0.01)  # Check every 10ms


async def test_evaluate_agent():
    # Initialize tracking variables
    max_size_tracker = {"max": 0}
    stop_event = threading.Event()
    monitor_thread = None

    try:
        # Start the monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_list_size,
            args=(
                trace_manager.traces_to_evaluate,
                max_size_tracker,
                stop_event,
            ),
            daemon=True,
        )
        monitor_thread.start()

        for golden in dataset.evals_iterator():
            task = asyncio.create_task(agent.run(golden.input))
            dataset.evaluate(task)
    finally:
        # Stop monitoring
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=1.0)

        assert max_size_tracker["max"] == len(dataset.goldens)
        test_exporter.clear_span_json_list()
