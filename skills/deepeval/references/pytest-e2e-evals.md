# Pytest End-to-End Evals

Use this for the default CI/CD path. End-to-end pytest evals call the app, build
test cases, and run `assert_test(test_case=..., metrics=...)`.

Do not use tracing primitives in the E2E template just to create an
`LLMTestCase`. Do not use `evals_iterator` inside pytest templates.

## Default Shape

Use `templates/test_single_turn_e2e.py` for single-turn E2E evals. This covers
plain LLM, RAG, and agent use cases by adapting `APP_RESPONSE_ADAPTER`.

```python
import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

DATASET_PATH = "tests/evals/.dataset.json"

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden):
    actual_output = your_llm_app(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=getattr(golden, "expected_output", None),
    )
    assert_test(test_case=test_case, metrics=END_TO_END_METRICS)
```

Run with:

```bash
deepeval test run tests/evals/test_<app>.py
```

Do not default to the raw `pytest` command.

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
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


async def chatbot_callback(input: str, turns=None, thread_id=None):
    response = await TARGET_APP_ENTRYPOINT(input, turns, thread_id)
    return Turn(role="assistant", content=APP_RESPONSE_ADAPTER(response))


simulator = ConversationSimulator(model_callback=chatbot_callback)
test_cases = simulator.simulate(
    conversational_goldens=dataset.goldens,
    max_user_simulations=MAX_TURNS,
)
```

Then parametrize over the simulated cases:

```python
@pytest.mark.parametrize("test_case", test_cases)
def test_conversation(test_case):
    assert_test(test_case=test_case, metrics=END_TO_END_METRICS)
```

## Python Script Fallback

Only create a Python script if the user pushes back on pytest. Explain that
pytest is preferred because it leaves a durable eval suite the user can rerun in
CI. If writing the fallback script, `evaluate()` or `evals_iterator` are
acceptable depending on the eval type.
