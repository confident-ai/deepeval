# Pytest End-to-End Evals

Use this for the default CI/CD path. End-to-end pytest evals run one golden
through the real app per test. If tracing or a supported integration is
available, pass the golden directly to DeepEval with
`assert_test(golden=golden, metrics=...)`.

Use `templates/test_single_turn_tracing.py` for the default single-turn path.
Use `templates/test_single_turn_no_tracing.py` only when the user explicitly
declines tracing or no tracing path is viable.

## Default Shape

Use an integration callback/instrumentation hook when one exists. If no native
integration exists, wrap the app entry point with `@observe` and update the
trace output.

```python
from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden

from metrics import SINGLE_TURN_TRACE_METRICS

ai_app = import_module("ai_app")


dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")

@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden: Golden):
    ai_app.run_traced_ai_app(golden.input)
    assert_test(golden=golden, metrics=SINGLE_TURN_TRACE_METRICS)
```

Run with:

```bash
deepeval test run tests/evals/test_<app>.py
```

Do not default to the raw `pytest` command.

## Integration-First Rule

Before using manual `@observe`, read `references/integrations.md` and the exact
integration doc for the app framework. For LangGraph, LangChain, OpenAI Agents,
Pydantic AI, CrewAI, Google ADK, Strands, AgentCore, and OpenTelemetry-backed
apps, the native integration should be the first implementation path.

For integration-backed pytest evals, the shape is still:

```python
@pytest.mark.parametrize("golden", dataset.goldens)
def test_agent(golden: Golden):
    run_ai_app_with_integration_tracing(golden.input)
    assert_test(golden=golden, metrics=SINGLE_TURN_TRACE_METRICS)
```

Do not translate these traced runs into `LLMTestCase`.

## Span Metrics In The Same Eval

Component-level metrics are part of the single-turn tracing eval. Do not create
a separate component test file. Attach span metrics at the component boundary
and keep `assert_test(golden=golden, ...)` at the trace level.

Use `next_*_span(metrics=[...])` when an integration creates the component span:

```python
from deepeval.tracing import next_retriever_span

from metrics import RETRIEVER_SPAN_METRICS


@pytest.mark.parametrize("golden", dataset.goldens)
def test_agent(golden: Golden):
    with next_retriever_span(metrics=RETRIEVER_SPAN_METRICS):
        run_ai_app_with_integration_tracing(golden.input)
    assert_test(golden=golden, metrics=SINGLE_TURN_TRACE_METRICS)
```

Use `@observe(metrics=[...])` when manually instrumenting the component or when
the integration supports observed component spans.

## No-Tracing Fallback

Only use the no-tracing template when tracing is intentionally out of scope. In
that case, a small wrapper around the AI app call is acceptable because this
path constructs the minimal `LLMTestCase` from AI app output and golden
reference fields before calling `assert_test(test_case=..., metrics=...)`.

## Useful `deepeval test run` Flags

Check available flags when unsure:

```bash
deepeval test run --help
```

Use these frequently:

| Flag | Use when |
| --- | --- |
| `--identifier`, `-id` | Label the run with useful context, for example `iterating-on-retrieval-round-1` or `iterating-on-tool-use-round-2`. |
| `--num-processes`, `-n` | Speed up large eval suites with pytest-xdist workers. Start around `-n 5` on modest machines and `-n 10` on stronger machines. |
| `--ignore-errors`, `-i` | Continue the run when individual DeepEval evaluation errors occur. Useful for large datasets. |
| `--skip-on-missing-params`, `-s` | Skip test cases missing fields required by a metric instead of failing the whole run. Useful when datasets are large or partly incomplete. |
| `--display`, `-d` | Control how much result detail is shown. Use when output is too noisy. |

For first runs on non-trivial datasets, a good starting command is:

```bash
deepeval test run tests/evals/test_<app>.py \
  --identifier "iterating-on-<purpose>-round-1" \
  --num-processes 5 \
  --ignore-errors \
  --skip-on-missing-params
```

Use purpose-based identifiers because they are easier to scan locally and look
better in Confident AI reports. Keep them short and kebab-case.

Increase `--num-processes` only if the user's machine and model provider limits
can handle more concurrency.

## Conversation E2E

For chatbot / multi-turn agent use cases, use `templates/test_multi_turn_e2e.py`. It
must simulate conversational test cases after loading the dataset, then
parametrize over the simulated test cases.

Multi-turn end-to-end evals must use multi-turn conversational metrics such as
`ConversationCompletenessMetric`, `RoleAdherenceMetric`, `TurnRelevancyMetric`,
or `ConversationalGEval`. Do not use single-turn `LLMTestCase` metrics for
multi-turn evals.

The minimal shape is:

```python
from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.simulator import ConversationSimulator

from metrics import MULTI_TURN_METRICS

MAX_TURNS = 10
ai_app = import_module("ai_app")

simulator = ConversationSimulator(model_callback=ai_app.chatbot_callback)
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")

@pytest.mark.parametrize(
    "test_case",
    simulator.simulate(
        conversational_goldens=dataset.goldens,
        max_user_simulations=MAX_TURNS,
    ),
)
def test_conversation(test_case):
    assert_test(test_case=test_case, metrics=MULTI_TURN_METRICS)
```

## Python Script Fallback

Only create a Python script if the user pushes back on pytest. Explain that
pytest is preferred because it leaves a durable eval suite the user can rerun in
CI. For traced single-turn scripts, use `evals_iterator` with goldens:

```python
for golden in dataset.evals_iterator(metrics=SINGLE_TURN_TRACE_METRICS):
    run_ai_app_with_integration_tracing(golden.input)
```

Use `evaluate()` only when it is a better fit for an already-built list of test
cases.
