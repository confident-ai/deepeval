"""
Post-hoc evaluation using LLMTestCase.trace_dict
-------------------------------------------------
DeepEval's agentic trace metrics (TaskCompletionMetric, StepEfficiencyMetric,
PlanQualityMetric, PlanAdherenceMetric) work with two approaches:

APPROACH 1 — @observe at runtime (standard DeepEval way)
  Instrument functions with @observe before they run. DeepEval captures the
  trace automatically and writes it into LLMTestCase.trace_dict after the call.

APPROACH 2 — post-hoc via trace_dict (this file)
  Your agent has *already* run and you have a saved trace (from a log file,
  database, observability system, or a previous @observe run that was
  serialised). Pass it directly as trace_dict= when constructing LLMTestCase.
  No @observe, no live agent execution required.

This is useful for:
  - Agents you don't own or can't re-run (3rd-party pipelines)
  - Offline / batch evaluation from logs
  - CI workflows that replay saved traces
  - Post-mortem analysis of production runs

The trace dict shape must match what TraceManager.create_nested_spans_dict()
produces:

  {
    "name":     str,
    "type":     "agent" | "tool" | "llm" | "retriever" | "custom",
    "input":    {...},
    "output":   {...},
    "children": [ <same shape, recursively> ]
  }

Run:
  python examples/tracing/test_posthoc_evaluation.py
"""

import json

from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# 1. A saved trace (in practice you'd load this from a JSON file / DB / etc.)
# ---------------------------------------------------------------------------
def load_saved_trace() -> dict:
    """
    Simulates loading a pre-recorded agent trace.
    In real usage: json.load(open("traces/run_2024-01-15.json"))
    """
    return {
        "name": "trip_planner_agent",
        "type": "agent",
        "input": {"input": "Plan a 2-day trip to Paris with restaurants."},
        "output": {
            "output": [
                "Eiffel Tower",
                "Louvre Museum",
                "Le Jules Verne",
                "Angelina Paris",
                "Septime",
            ]
        },
        "children": [
            {
                "name": "itinerary_generator",
                "type": "tool",
                "input": {
                    "inputParameters": {"destination": "Paris", "days": 2}
                },
                "output": {"output": ["Eiffel Tower", "Louvre Museum"]},
                "children": [],
            },
            {
                "name": "restaurant_finder",
                "type": "tool",
                "input": {"inputParameters": {"city": "Paris"}},
                "output": {
                    "output": ["Le Jules Verne", "Angelina Paris", "Septime"]
                },
                "children": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# 2. Build LLMTestCase with the saved trace — no @observe needed
# ---------------------------------------------------------------------------
saved_trace = load_saved_trace()

test_case = LLMTestCase(
    input="Plan a 2-day trip to Paris with restaurants.",
    actual_output=(
        "Eiffel Tower, Louvre Museum, Le Jules Verne, Angelina Paris, Septime"
    ),
    trace_dict=saved_trace,  # post-hoc trace injection
)

# trace_dict is also accepted via its camelCase alias for JSON round-trips:
#   LLMTestCase.model_validate({"input": "...", "traceDict": saved_trace})
assert test_case.trace_dict == saved_trace

# ---------------------------------------------------------------------------
# 3. Evaluate — TaskCompletionMetric detects trace_dict and uses the
#    trace-aware prompt path (same as it would with @observe at runtime)
# ---------------------------------------------------------------------------
task_completion = TaskCompletionMetric(threshold=0.5)

print("Running post-hoc evaluation with saved trace …")
evaluate([test_case], [task_completion])

print(f"\nScore : {task_completion.score}")
print(f"Reason: {task_completion.reason}")
