# Pytest Component Evals

Use this only when a specific component needs span-level diagnostics: retriever,
generator, tool, planner, or another internal step.

Component-level evals are single-turn only. There is no multi-turn component
level: multi-turn evals evaluate the conversation as a whole with
`ConversationalTestCase`s and multi-turn metrics.

Component evals are a superset of an E2E trace. In tracing, the trace is the
end-to-end execution and spans are the components. Span-level metrics evaluate
specific spans inside the trace, while the trace itself still represents the
full E2E run.

Component evals are separate from end-to-end `LLMTestCase` tests. Do not mix the
two styles in one pytest function.

Component-level evals are an add-on to E2E, not a replacement. If component
metrics are needed, keep the E2E test file and add
`templates/test_single_turn_component.py` only for the specific span that needs
diagnostics.

## Pattern

Attach metrics to the observed component span, update the span test case, then
assert the active trace with the golden:

```python
import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span

DATASET_PATH = "tests/evals/.dataset.json"
SPAN_LEVEL_METRICS = []

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


@observe(metrics=SPAN_LEVEL_METRICS)
def observed_component(user_input: str):
    actual_output = component(user_input)
    update_current_span(
        test_case=LLMTestCase(input=user_input, actual_output=actual_output)
    )
    return actual_output


@pytest.mark.parametrize("golden", dataset.goldens)
def test_single_turn_component(golden):
    observed_component(golden.input)
    assert_test(golden=golden)
```

Run with:

```bash
deepeval test run tests/evals/test_single_turn_component.py
```

## When to Add

Add component evals when end-to-end failures are hard to debug or when the user
explicitly wants to evaluate a component in isolation.

Examples:

- retriever contextual relevancy
- generator answer relevancy
- tool correctness
- planner or step quality

If end-to-end metrics answer the question, do not add span-level metrics just to
add tracing.
